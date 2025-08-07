import requests
import json
import numpy as np
import time

class SeizureDetectionClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        
    def test_connection(self):
        """Test if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ API is running!")
                print(f"Status: {data['status']}")
                print(f"Model loaded: {data['model_loaded']}")
                return True
            else:
                print("❌ API is not responding correctly")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Make sure the Flask app is running.")
            return False
    
    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            response = requests.get(f"{self.base_url}/model_info")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting model info: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def predict(self, features):
        """Make a prediction using the API"""
        try:
            payload = {"features": features}
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                print(f"Prediction error: {error_data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def generate_sample_data(self, feature_count):
        """Generate random sample data for testing"""
        return np.random.randn(feature_count).tolist()
    
    def run_test_suite(self):
        """Run a comprehensive test of the API"""
        print("🧪 Starting API Test Suite\n")
        
        # Test 1: Connection
        print("Test 1: API Connection")
        if not self.test_connection():
            return
        print()
        
        # Test 2: Model Info
        print("Test 2: Model Information")
        model_info = self.get_model_info()
        if model_info:
            print(f"✅ Model loaded successfully")
            print(f"Features expected: {model_info['feature_count']}")
            print(f"Classes: {model_info['classes']}")
            print(f"Model accuracy: {model_info['model_metrics']['accuracy']:.4f}")
            feature_count = model_info['feature_count']
        else:
            print("❌ Failed to get model info")
            return
        print()
        
        # Test 3: Single Prediction
        print("Test 3: Single Prediction")
        sample_data = self.generate_sample_data(feature_count)
        result = self.predict(sample_data)
        
        if result:
            print("✅ Prediction successful!")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Timestamp: {result['timestamp']}")
            
            # Display class probabilities
            print("Class probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  Class {class_name}: {prob:.4f}")
        else:
            print("❌ Prediction failed")
            return
        print()
        
        # Test 4: Multiple Predictions (Performance Test)
        print("Test 4: Performance Test (10 predictions)")
        start_time = time.time()
        successful_predictions = 0
        
        for i in range(10):
            sample_data = self.generate_sample_data(feature_count)
            result = self.predict(sample_data)
            if result:
                successful_predictions += 1
            print(f"  Prediction {i+1}/10: {'✅' if result else '❌'}")
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10
        
        print(f"\nPerformance Results:")
        print(f"Successful predictions: {successful_predictions}/10")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per prediction: {avg_time:.3f} seconds")
        print()
        
        # Test 5: Edge Cases
        print("Test 5: Edge Cases")
        
        # Test with all zeros
        print("  Testing with all zeros...")
        zero_data = [0.0] * feature_count
        result = self.predict(zero_data)
        print(f"  All zeros: {'✅' if result else '❌'}")
        
        # Test with very large values
        print("  Testing with large values...")
        large_data = [1000.0] * feature_count
        result = self.predict(large_data)
        print(f"  Large values: {'✅' if result else '❌'}")
        
        # Test with very small values
        print("  Testing with small values...")
        small_data = [0.001] * feature_count
        result = self.predict(small_data)
        print(f"  Small values: {'✅' if result else '❌'}")
        
        # Test with wrong number of features
        print("  Testing with wrong feature count...")
        wrong_data = [1.0] * (feature_count - 1)  # One less feature
        result = self.predict(wrong_data)
        print(f"  Wrong feature count: {'✅ (correctly rejected)' if not result else '❌ (should have failed)'}")
        
        print("\n🎉 Test suite completed!")

def main():
    """Main function to run the test client"""
    print("Epileptic Seizure Detection API Test Client")
    print("=" * 50)
    
    # Initialize client
    client = SeizureDetectionClient()
    
    # Run test suite
    client.run_test_suite()
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode - Enter 'quit' to exit")
    
    model_info = client.get_model_info()
    if not model_info:
        print("❌ Cannot get model info. Exiting.")
        return
    
    feature_count = model_info['feature_count']
    
    while True:
        print(f"\nOptions:")
        print("1. Make prediction with random data")
        print("2. Make prediction with custom data")
        print("3. Get model info")
        print("4. Test connection")
        print("5. Quit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("Generating random sample data...")
            sample_data = client.generate_sample_data(feature_count)
            result = client.predict(sample_data)
            
            if result:
                seizure_status = "SEIZURE DETECTED" if result['predicted_class'] == 1 else "NO SEIZURE"
                print(f"\n🔍 Prediction Result: {seizure_status}")
                print(f"Confidence: {result['confidence']:.2%}")
            else:
                print("❌ Prediction failed")
                
        elif choice == "2":
            print(f"Enter {feature_count} comma-separated values:")
            try:
                user_input = input("Features: ")
                features = [float(x.strip()) for x in user_input.split(",")]
                
                if len(features) != feature_count:
                    print(f"❌ Expected {feature_count} features, got {len(features)}")
                    continue
                
                result = client.predict(features)
                if result:
                    seizure_status = "SEIZURE DETECTED" if result['predicted_class'] == 1 else "NO SEIZURE"
                    print(f"\n🔍 Prediction Result: {seizure_status}")
                    print(f"Confidence: {result['confidence']:.2%}")
                else:
                    print("❌ Prediction failed")
                    
            except ValueError:
                print("❌ Invalid input. Please enter numeric values separated by commas.")
                
        elif choice == "3":
            info = client.get_model_info()
            if info:
                print(f"\n📊 Model Information:")
                print(f"Feature count: {info['feature_count']}")
                print(f"Classes: {info['classes']}")
                print(f"Accuracy: {info['model_metrics']['accuracy']:.4f}")
                print(f"F1 Score: {info['model_metrics']['f1_score']:.4f}")
            
        elif choice == "4":
            client.test_connection()
            
        elif choice == "5" or choice.lower() == "quit":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()