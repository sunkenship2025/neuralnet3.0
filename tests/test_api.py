"""
Test client for IoT Device Identification API
Demonstrates how to use the API with Python requests
"""

import requests
import json
import pandas as pd
from pathlib import Path


def test_health_check(api_url: str = "http://localhost:8000"):
    """Test API health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{api_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_get_features(api_url: str = "http://localhost:8000", top_k: int = 10):
    """Test get features endpoint"""
    print("\n" + "="*60)
    print(f"Testing Get Top {top_k} Features")
    print("="*60)
    
    response = requests.get(f"{api_url}/features?top_k={top_k}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Top {top_k} Features:")
        for i, feat in enumerate(data['features'], 1):
            print(f"  {i}. {feat}")
        return data
    else:
        print(f"Error: {response.text}")
        return None


def test_predict_json(api_url: str = "http://localhost:8000", model_type: str = "cnn_lstm"):
    """Test prediction with JSON payload"""
    print("\n" + "="*60)
    print(f"Testing JSON Prediction (Model: {model_type})")
    print("="*60)
    
    # Load sample data
    try:
        test_df = pd.read_csv("Aalto_test_IoTDevID (1).csv")
        veto_df = pd.read_csv("veto_average_results (1).csv")
        
        # Get top features
        top_features = veto_df.nlargest(50, 'Average')['Feature'].tolist()
        
        # Create sample from first 3 rows
        sample_features = test_df[top_features].head(3).to_dict('records')
        
        # Make request
        payload = {
            "features": sample_features,
            "model_type": model_type,
            "top_k_features": 50
        }
        
        response = requests.post(f"{api_url}/predict", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nPredictions:")
            for i, (pred, probs) in enumerate(zip(data['predictions'], data['probabilities']), 1):
                print(f"\n  Sample {i}: {pred}")
                print(f"    Confidence: {probs.get(pred, 0):.2%}")
                print(f"    Top alternatives:")
                for device, prob in list(probs.items())[:3]:
                    print(f"      - {device}: {prob:.2%}")
            return data
        else:
            print(f"Error: {response.text}")
            return None
            
    except FileNotFoundError as e:
        print(f"Error: Required data files not found: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_predict_csv(api_url: str = "http://localhost:8000", model_type: str = "cnn_lstm"):
    """Test prediction with CSV upload"""
    print("\n" + "="*60)
    print(f"Testing CSV Upload Prediction (Model: {model_type})")
    print("="*60)
    
    csv_path = "Aalto_test_IoTDevID (1).csv"
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        return None
    
    try:
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            params = {'model_type': model_type}
            
            response = requests.post(
                f"{api_url}/predict/csv",
                files=files,
                params=params
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResults:")
            print(f"  Samples processed: {data['num_samples']}")
            print(f"  Average confidence: {data['avg_confidence']:.2%}")
            print(f"  Confidence range: {data['min_confidence']:.2%} - {data['max_confidence']:.2%}")
            print(f"\n  Class Distribution:")
            for device, count in sorted(data['class_distribution'].items(), key=lambda x: x[1], reverse=True):
                pct = count / data['num_samples'] * 100
                print(f"    - {device}: {count} ({pct:.1f}%)")
            return data
        else:
            print(f"Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_model_comparison(api_url: str = "http://localhost:8000"):
    """Compare predictions across different models"""
    print("\n" + "="*60)
    print("Testing Model Comparison")
    print("="*60)
    
    models = ["cnn_lstm", "hgb", "lstm"]
    results = {}
    
    try:
        test_df = pd.read_csv("Aalto_test_IoTDevID (1).csv")
        veto_df = pd.read_csv("veto_average_results (1).csv")
        top_features = veto_df.nlargest(50, 'Average')['Feature'].tolist()
        
        # Get single sample
        sample = test_df[top_features].head(1).to_dict('records')
        
        print("\nComparing models on same sample...")
        for model in models:
            try:
                payload = {
                    "features": sample,
                    "model_type": model,
                    "top_k_features": 50
                }
                
                response = requests.post(f"{api_url}/predict", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    pred = data['predictions'][0]
                    conf = max(data['probabilities'][0].values())
                    results[model] = {'prediction': pred, 'confidence': conf}
                    print(f"  {model:15s}: {pred:25s} (confidence: {conf:.2%})")
                else:
                    print(f"  {model:15s}: Error - Model not exported yet")
            except Exception as e:
                print(f"  {model:15s}: Error - {e}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Run all API tests"""
    API_URL = "http://localhost:8000"
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "IoT Device API Test Suite" + " "*17 + "║")
    print("╚" + "="*58 + "╝")
    
    # Check if API is running
    try:
        response = requests.get(API_URL, timeout=2)
        if response.status_code != 200:
            print(f"\n⚠️  Warning: API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Cannot connect to API at {API_URL}")
        print("   Make sure the API is running:")
        print("   $ python api.py")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_check(API_URL)),
        ("Get Features", lambda: test_get_features(API_URL, top_k=10)),
        ("JSON Prediction", lambda: test_predict_json(API_URL, "cnn_lstm")),
        ("CSV Upload", lambda: test_predict_csv(API_URL, "cnn_lstm")),
        ("Model Comparison", lambda: test_model_comparison(API_URL))
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ Passed" if result else "⚠️  Warning"
        except Exception as e:
            results[test_name] = f"❌ Failed: {str(e)}"
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, status in results.items():
        print(f"  {test_name:25s}: {status}")
    
    print("\n" + "="*60)
    print("✨ Testing complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
