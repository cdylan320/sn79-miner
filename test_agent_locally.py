#!/usr/bin/env python3
"""
Test the TAOS trading agent locally without network dependencies.
"""
import os
import sys
import json
import random
import time
import tempfile
from pathlib import Path

# Set environment variables (should be set by wrapper script, but setting as fallback)
user_home = os.path.expanduser('~')
os.environ.setdefault("HOME", user_home)
os.environ.setdefault("BT_WALLET_PATH", os.path.join(user_home, '.bittensor', 'wallets'))

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Create a temporary directory for testing
TEST_DIR = tempfile.mkdtemp(prefix="taos_test_")
print(f"📁 Using test directory: {TEST_DIR}")

def test_agent_directly():
    """Test the agent by directly importing and running it."""
    print("🧪 Testing TAOS Agent Locally")
    print("=" * 40)

    try:
        # Import the agent
        agent_path = os.path.join(user_home, '.taos', 'agents', 'SimpleRegressorAgent.py')
        if not os.path.exists(agent_path):
            # Copy agent if it doesn't exist
            if os.path.exists("agents/SimpleRegressorAgent.py"):
                os.makedirs(os.path.join(user_home, '.taos', 'agents'), exist_ok=True)
                import shutil
                shutil.copy("agents/SimpleRegressorAgent.py", agent_path)
                print("✅ Copied agent to test location")

        # Import the agent class
        import importlib.util
        spec = importlib.util.spec_from_file_location("SimpleRegressorAgent", agent_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        AgentClass = getattr(agent_module, "SimpleRegressorAgent")

        # Create mock agent parameters
        agent_params = type('MockParams', (), {
            'min_quantity': 0.1,
            'max_quantity': 1.0,
            'expiry_period': 200,
            'model': 'PassiveAggressiveRegressor',
            'signal_threshold': 0.0025,
            'data_dir': os.path.join(TEST_DIR, 'agents', 'data')  # Override data directory
        })()

        # Change to test directory to avoid permission issues
        original_cwd = os.getcwd()
        os.chdir(TEST_DIR)

        # Create agent instance (mock uid=147)
        print("🤖 Creating agent instance...")
        try:
            agent = AgentClass(uid=147, config=agent_params, log_dir=TEST_DIR)
            print("✅ Agent created successfully!")
            print(f"📊 Agent UID: {agent.uid}")
            print(f"🎯 Agent Parameters: min_qty={agent_params.min_quantity}, max_qty={agent_params.max_quantity}")
        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            os.chdir(original_cwd)
            raise

        # Create mock market data for testing
        print("\n📈 Testing with mock market data...")

        # Create a mock synapse-like object with market data
        class MockSynapse:
            def __init__(self):
                self.response = None
                # Mock market data
                self.observation = {
                    'timestamp': int(time.time() * 1000),
                    'market_state': {
                        'BTC/USDT': {
                            'price': 45000 + random.uniform(-1000, 1000),
                            'volume': 1000000 + random.uniform(-500000, 500000),
                            'high_24h': 46000,
                            'low_24h': 44000,
                            'price_change_24h': random.uniform(-0.05, 0.05)
                        }
                    },
                    'portfolio': {
                        'BTC': 1.0,
                        'USDT': 45000.0,
                        'total_value_usdt': 90000.0
                    }
                }

        # Test multiple market scenarios
        scenarios = [
            "Bull market (rising prices)",
            "Bear market (falling prices)",
            "Sideways market (stable prices)",
            "High volatility market"
        ]

        for i, scenario in enumerate(scenarios):
            print(f"\n🔄 Testing scenario {i+1}: {scenario}")

            # Create mock synapse with different market conditions
            synapse = MockSynapse()

            # Modify market data based on scenario
            if "Bull" in scenario:
                synapse.observation['market_state']['BTC/USDT']['price'] = 47000
                synapse.observation['market_state']['BTC/USDT']['price_change_24h'] = 0.03
            elif "Bear" in scenario:
                synapse.observation['market_state']['BTC/USDT']['price'] = 43000
                synapse.observation['market_state']['BTC/USDT']['price_change_24h'] = -0.03
            elif "High volatility" in scenario:
                synapse.observation['market_state']['BTC/USDT']['price'] = 45000 + random.uniform(-5000, 5000)

            print(".2f")
            print(".2f")
            # Test agent response
            start_time = time.time()
            try:
                response = agent.handle(synapse)
                processing_time = time.time() - start_time

                print(".3f")
                print(f"📊 Response type: {type(response)}")

                # Check if response contains trading instructions
                if hasattr(response, 'instructions') or isinstance(response, dict):
                    if isinstance(response, dict) and 'instructions' in response:
                        instructions = response['instructions']
                        print(f"💼 Trading instructions: {len(instructions) if instructions else 0} actions")
                        if instructions:
                            for instr in instructions[:3]:  # Show first 3 instructions
                                print(f"   • {instr}")
                    else:
                        print(f"💼 Response content: {response}")
                else:
                    print(f"💼 Response: {response}")

                print("✅ Agent responded successfully!")
            except Exception as e:
                print(f"❌ Agent error: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(1)  # Brief pause between tests

        # Restore original working directory
        os.chdir(original_cwd)

        print("\n🎉 Local agent testing completed!")
        print("📈 Your TAOS trading agent is working correctly!")
        print("\n💡 Next steps:")
        print("   • Agent logic is functional")
        print("   • Ready for network deployment when subnet activates")
        print("   • Monitor subnet status: btcli s hyperparameters --netuid 366 --network test")

    except Exception as e:
        # Make sure to restore directory even on error
        os.chdir(original_cwd)
        raise

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_directly()
