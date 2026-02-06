import os
import sys
from typing import Optional, Callable, Any

# Try to import azure-identity (may not be available in all environments)
try:
    from azure.identity import (
        ChainedTokenCredential,
        AzureCliCredential,
        ManagedIdentityCredential,
        SharedTokenCacheCredential,
        get_bearer_token_provider
    )
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

# Try to import msal for direct token refresh (works without az CLI)
try:
    import msal
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False

# Azure CLI's public client ID (used for MSAL token refresh)
AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'

class MSALTokenProvider:
    """
    Token provider that uses MSAL to refresh tokens dynamically.
    Uses shared locks for reading to support high parallelism.
    """

    def __init__(self, scope: str = 'api://trapi/.default'):
        self.scope = scope
        self.cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
        self._last_token_time = None
        self._token_refresh_count = 0
        self._lock_held = False
        self.cache = msal.SerializableTokenCache()
        self.app = msal.PublicClientApplication(
            AZURE_CLI_CLIENT_ID,
            authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
            token_cache=self.cache,
        )
        self._load_cache()

    def _load_cache(self) -> bool:
        """Load the MSAL cache from disk."""
        if not MSAL_AVAILABLE:
            return False
        if not os.path.exists(self.cache_path):
            return False
        
        if self._lock_held:
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache.deserialize(f.read())
                return True
            except Exception:
                return False

        lock_path = self.cache_path + ".lock"
        try:
            import fcntl
            with open(lock_path, 'a+') as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_SH)
                try:
                    with open(self.cache_path, 'r') as f:
                        self.cache.deserialize(f.read())
                    return True
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception as e:
            # print(f"[MSALTokenProvider] Failed to load cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save the MSAL cache to disk if it has changed."""
        if self.cache.has_state_changed:
            if self._lock_held:
                try:
                    with open(self.cache_path, 'w') as f:
                        f.write(self.cache.serialize())
                    return
                except Exception:
                    return

            lock_path = self.cache_path + ".lock"
            try:
                import fcntl
                with open(lock_path, 'a+') as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    try:
                        with open(self.cache_path, 'w') as f:
                            f.write(self.cache.serialize())
                        # print(f"[MSALTokenProvider] Cache persisted to disk")
                    finally:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception as e:
                pass
                # print(f"[MSALTokenProvider] Failed to save cache: {e}")

    def _find_valid_at_in_cache(self, accounts):
        import time
        now = time.time() + 300
        for account in accounts:
            try:
                for token in self.cache.find(msal.TokenCache.CredentialType.ACCESS_TOKEN, 
                                           query={"home_account_id": account["home_account_id"]}):
                    if self.scope in token.get("target", ""):
                        expires_on = int(token.get("expires_on", 0))
                        if expires_on > now:
                            return token.get("secret")
            except Exception:
                pass
        return None

    def __call__(self) -> str:
        """Get a fresh access token, refreshing if necessary."""
        import time
        import fcntl

        if not MSAL_AVAILABLE:
            raise RuntimeError("MSAL not available - install msal package")

        accounts = self.app.get_accounts()
        if not accounts and os.path.exists(self.cache_path):
             self._load_cache()
             accounts = self.app.get_accounts()

        if not accounts:
            if not os.path.exists(self.cache_path):
                 raise RuntimeError(f"MSAL cache not found at {self.cache_path}")

        # 1. Fast Path
        token = self._find_valid_at_in_cache(accounts)
        if token:
            return token

        # print(f"[MSALTokenProvider] Fast path miss - waiting for lock...")
        start_wait = time.time()

        # 2. Slow Path
        lock_path = self.cache_path + ".lock"
        with open(lock_path, 'a+') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            self._lock_held = True
            try:
                wait_time = time.time() - start_wait
                # if wait_time > 1.0:
                #     print(f"[MSALTokenProvider] Lock acquired (waited {wait_time:.2f}s)")
                
                self._load_cache()
                accounts = self.app.get_accounts()
                
                if not accounts:
                    raise RuntimeError("No accounts found in MSAL cache. Run 'az login' first.")

                token = self._find_valid_at_in_cache(accounts)
                if token:
                    # print(f"[MSALTokenProvider] Token found after reload")
                    return token

                # 3. Refresh
                last_error = None
                for i, account in enumerate(accounts):
                    # username = account.get('username', 'unknown')
                    result = self.app.acquire_token_silent([self.scope], account=account)

                    if result and 'access_token' in result:
                        self._save_cache()
                        self._token_refresh_count += 1
                        self._last_token_time = time.time()

                        # if self._token_refresh_count == 1 or self._token_refresh_count % 100 == 0:
                        #     print(f"[MSALTokenProvider] Token refreshed (count: {self._token_refresh_count}, account: {username})")

                        return result['access_token']
                    if result:
                        last_error = f"{result.get('error', 'unknown')}: {result.get('error_description', '')}"

                raise RuntimeError(f"Token refresh failed: {last_error}")
            finally:
                self._lock_held = False
                fcntl.flock(lock_file, fcntl.LOCK_UN)


        # All accounts failed
        account_names = [a.get('username', 'unknown') for a in accounts]
        raise RuntimeError(f"Token acquisition failed for all {len(accounts)} accounts ({', '.join(account_names)}). Last error: {last_error}")

def get_azure_token_provider(scope: str = 'api://trapi/.default') -> Optional[Callable[[], str]]:
    """
    Get a token provider for Azure OpenAI.
    Prioritizes MSALTokenProvider (if cache exists) for robustness,
    then falls back to azure-identity.
    """
    # Check for MSAL availability and cache
    if MSAL_AVAILABLE:
        cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
        if os.path.exists(cache_path):
            try:
                # print(f"[Azure] Initializing MSALTokenProvider with scope: {scope}")
                provider = MSALTokenProvider(scope=scope)
                # Quick test (optional, can be removed if performance is key)
                # provider() 
                return provider
            except Exception as e:
                print(f"[Azure] Failed to initialize MSALTokenProvider: {e}")

    # Fallback to Azure Identity
    if AZURE_IDENTITY_AVAILABLE:
        # print("[Azure] Falling back to azure-identity")
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        return get_bearer_token_provider(credential, scope)
    
    return None
