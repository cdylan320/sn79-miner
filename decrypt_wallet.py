#!/usr/bin/env python3
import json
import getpass
import os

# Auto-load miner.env file BEFORE importing bittensor
env_file = os.path.join(os.path.dirname(__file__), 'miner.env')
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                # Handle quoted values
                if line.count('=') == 1:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('\'"')
                    os.environ[key] = value
    print("✅ Loaded miner.env environment variables")
else:
    print("⚠️  miner.env file not found")

# Set additional environment variables needed for bittensor
# Use current user's home directory
user_home = os.path.expanduser('~')
os.environ['HOME'] = user_home
os.environ['BT_WALLET_PATH'] = os.path.join(user_home, '.bittensor', 'wallets')

# Now import bittensor after environment is set
import bittensor as bt

def decrypt_and_import_wallet():
    # Load the JSON wallet as string (bittensor expects JSON string, not dict)
    with open('bittensor.json', 'r') as f:
        wallet_json_str = f.read()
    wallet_data = json.loads(wallet_json_str)  # Parse for display purposes
    
    print(f"Wallet name: {wallet_data['meta']['name']}")
    print(f"Address: {wallet_data['address']}")
    print()
    
    # Get password from environment variable (should be set from miner.env)
    password = os.environ.get('JSON_PASSWORD')
    if not password:
        print("❌ Please source the miner.env file first:")
        print("   source miner.env")
        print("   Or set: export JSON_PASSWORD='your_password_here'")
        return False
    
    try:
        # Decrypt the wallet
        keypair = bt.Keypair.create_from_encrypted_json(wallet_json_str, password)
        print("✅ Successfully decrypted wallet!")
        print(f"SS58 Address: {keypair.ss58_address}")
        
        # Verify this matches the registered address
        if keypair.ss58_address == '5DJCeqFEQ59XhDK4kfxssE8jnwK3Y3Tq36SBphc1ufc6FjWf':
            print("✅ Address matches registered UID 147!")
            
            # Now create Bittensor wallet files
            # This is the coldkey, so we'll create it as the coldkey
            
            # Create the wallet directory structure
            wallet_dir = os.path.join(user_home, '.bittensor', 'wallets', 'cold_draven')
            hotkey_dir = os.path.join(wallet_dir, 'hotkeys')
            os.makedirs(wallet_dir, exist_ok=True)
            os.makedirs(hotkey_dir, exist_ok=True)
            
            # Create Bittensor wallet and set the coldkey directly from the decrypted keypair
            wallet = bt.Wallet('cold_draven')
            wallet.set_coldkey(keypair, encrypt=False, overwrite=True)

            # Also set the hotkey (same key since user uses same key for both)
            wallet.set_hotkey(keypair, encrypt=False, overwrite=True)
            
            print("✅ Saved wallet as Bittensor coldkey and hotkey!")
            print("🔄 Testing wallet load...")
            
            # Test loading the wallet
            test_wallet = bt.Wallet('cold_draven', 'default')
            print(f"✅ Wallet loaded! Coldkey: {test_wallet.coldkey.ss58_address}, Hotkey: {test_wallet.hotkey.ss58_address}")
            
            return True
        else:
            print("❌ Address doesn't match registered wallet!")
            print(f"Expected: 5DJCeqFEQ59XhDK4kfxssE8jnwK3Y3Tq36SBphc1ufc6FjWf")
            print(f"Got: {keypair.ss58_address}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to decrypt wallet: {e}")
        return False

if __name__ == "__main__":
    success = decrypt_and_import_wallet()
    if success:
        print("\n🎉 Wallet imported successfully!")
        print("You can now run your miner with UID 147!")
    else:
        print("\n❌ Wallet import failed!")
