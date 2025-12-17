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
os.environ['HOME'] = '/home/ocean'
os.environ['BT_WALLET_PATH'] = '/home/ocean/.bittensor/wallets'

# Now import bittensor after environment is set
from substrateinterface import Keypair
import bittensor as bt

def decrypt_and_import_wallet():
    # Load the JSON wallet
    with open('bittensor.json', 'r') as f:
        wallet_data = json.load(f)
    
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
        keypair = Keypair.create_from_encrypted_json(wallet_data, password)
        print("✅ Successfully decrypted wallet!")
        print(f"SS58 Address: {keypair.ss58_address}")
        
        # Verify this matches the registered address
        if keypair.ss58_address == '5DJCeqFEQ59XhDK4kfxssE8jnwK3Y3Tq36SBphc1ufc6FjWf':
            print("✅ Address matches registered UID 147!")
            
            # Now create Bittensor wallet files
            # This is the coldkey, so we'll create it as the coldkey

            # Create the wallet directory structure
            wallet_dir = '/home/ocean/.bittensor/wallets/cold_draven'
            hotkey_dir = os.path.join(wallet_dir, 'hotkeys')
            os.makedirs(wallet_dir, exist_ok=True)
            os.makedirs(hotkey_dir, exist_ok=True)

            # Create Bittensor keypair from the private key
            private_key_hex = keypair.private_key.hex()
            bt_keypair = bt.Keypair.create_from_private_key(private_key_hex)

            # Create Bittensor wallet and set the coldkey
            wallet = bt.Wallet('cold_draven')
            wallet.set_coldkey(bt_keypair, encrypt=False, overwrite=True)

            # Also set the hotkey (same key since user uses same key for both)
            wallet.set_hotkey(bt_keypair, encrypt=False, overwrite=True)

            print("✅ Saved wallet as Bittensor coldkey and hotkey!")
            print("🔄 Testing wallet load...")

            # Test loading the wallet
            test_wallet = bt.Wallet('cold_draven', 'miner')
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
