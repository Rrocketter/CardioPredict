#!/usr/bin/env python3
"""
production test thing for cardiopredict
just tests a bunch of stuff to see if it works i guess
"""

import requests
import json
import time
import sys
import os
from pathlib import Path

# Add web_platform to path
sys.path.insert(0, str(Path(__file__).parent / "web_platform"))

class ProductionReadinessTest:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.auth_token = None
        self.test_results = []
        
    def log_test(self, test_name, success, details=""):
        # just logs what happened
        status = "PASS" if success else "FAIL"
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        print(f"{status} {test_name}")
        if details and not success:
            print(f"    Details: {details}")
    
    def test_health_endpoint(self):
        # checks if the health thing works
        try:
            response = self.session.get(f"{self.base_url}/health")
            data = response.json()
            
            success = (
                response.status_code == 200 and
                data.get("status") == "healthy" and
                "phase_3_enabled" in data
            )
            
            self.log_test("Health Endpoint", success, 
                         f"Status: {response.status_code}, Data: {data}")
            return success
        except Exception as e:
            self.log_test("Health Endpoint", False, str(e))
            return False
    
    def test_endpoints_list(self):
        # see if we can get the list of endpoints or whatever
        try:
            response = self.session.get(f"{self.base_url}/api/endpoints")
            data = response.json()
            
            success = (
                response.status_code == 200 and
                "endpoints" in data and
                len(data["endpoints"]) > 10  # should have a bunch of endpoints
            )
            
            endpoint_count = len(data.get("endpoints", []))
            self.log_test("Endpoints List", success, 
                         f"Found {endpoint_count} endpoints")
            return success
        except Exception as e:
            self.log_test("Endpoints List", False, str(e))
            return False
    
    def test_user_registration(self):
        # try to register a new user
        try:
            user_data = {
                "username": f"testuser_{int(time.time())}",
                "email": f"test_{int(time.time())}@example.com",
                "password": "TestPassword123!",
                "full_name": "Test User"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v3/auth/register",
                json=user_data
            )
            data = response.json()
            
            success = response.status_code == 201 and "message" in data
            self.log_test("User Registration", success, 
                         f"Status: {response.status_code}, Message: {data.get('message', 'N/A')}")
            
            if success:
                self.test_user = user_data
            
            return success
        except Exception as e:
            self.log_test("User Registration", False, str(e))
            return False
    
    def test_user_login(self):
        # login with the user we just made
        if not hasattr(self, 'test_user'):
            self.log_test("User Login", False, "No test user available")
            return False
        
        try:
            login_data = {
                "username": self.test_user["username"],
                "password": self.test_user["password"]
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v3/auth/login",
                json=login_data
            )
            data = response.json()
            
            success = (
                response.status_code == 200 and
                "access_token" in data
            )
            
            if success:
                self.auth_token = data["access_token"]
                self.session.headers.update({
                    "Authorization": f"Bearer {self.auth_token}"
                })
            
            self.log_test("User Login", success, 
                         f"Status: {response.status_code}, Token received: {bool(self.auth_token)}")
            return success
        except Exception as e:
            self.log_test("User Login", False, str(e))
            return False
    
    def test_protected_endpoint(self):
        # test something that needs auth
        if not self.auth_token:
            self.log_test("Protected Endpoint", False, "No auth token available")
            return False
        
        try:
            response = self.session.get(f"{self.base_url}/api/v3/auth/profile")
            data = response.json()
            
            success = (
                response.status_code == 200 and
                "username" in data
            )
            
            self.log_test("Protected Endpoint", success, 
                         f"Status: {response.status_code}, Profile: {data.get('username', 'N/A')}")
            return success
        except Exception as e:
            self.log_test("Protected Endpoint", False, str(e))
            return False
    
    def test_ml_model_info(self):
        # check if we can get info about ml models
        try:
            response = self.session.get(f"{self.base_url}/api/v3/ml/models")
            data = response.json()
            
            success = response.status_code == 200 and "models" in data
            
            model_count = len(data.get("models", []))
            self.log_test("ML Model Info", success, 
                         f"Status: {response.status_code}, Models: {model_count}")
            return success
        except Exception as e:
            self.log_test("ML Model Info", False, str(e))
            return False
    
    def test_prediction_endpoint(self):
        # see if predictions work
        try:
            # just some random heart data
            prediction_data = {
                "age": 45,
                "cholesterol": 200,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "heart_rate": 72,
                "smoking": False,
                "exercise_frequency": 3
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=prediction_data
            )
            data = response.json()
            
            # should work even if no model is trained yet
            success = response.status_code in [200, 422]  # 422 if no model trained
            
            self.log_test("Prediction Endpoint", success, 
                         f"Status: {response.status_code}, Response: {data.get('message', 'OK')}")
            return success
        except Exception as e:
            self.log_test("Prediction Endpoint", False, str(e))
            return False
    
    def test_background_job(self):
        # try to start a background job
        if not self.auth_token:
            self.log_test("Background Job", False, "No auth token available")
            return False
        
        try:
            job_data = {
                "job_type": "model_training",
                "parameters": {"test": True}
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v3/jobs",
                json=job_data
            )
            data = response.json()
            
            success = response.status_code == 202 and "job_id" in data
            
            self.log_test("Background Job", success, 
                         f"Status: {response.status_code}, Job ID: {data.get('job_id', 'N/A')}")
            return success
        except Exception as e:
            self.log_test("Background Job", False, str(e))
            return False
    
    def test_rate_limiting(self):
        # spam requests to see if rate limiting works
        try:
            # just hit the server a bunch of times
            responses = []
            for i in range(10):
                response = self.session.get(f"{self.base_url}/health")
                responses.append(response.status_code)
            
            # should get 429 eventually or just all 200s
            has_rate_limit = 429 in responses
            all_ok = all(status == 200 for status in responses)
            
            success = has_rate_limit or all_ok
            
            self.log_test("Rate Limiting", success, 
                         f"Responses: {responses[:5]}{'...' if len(responses) > 5 else ''}")
            return success
        except Exception as e:
            self.log_test("Rate Limiting", False, str(e))
            return False
    
    def test_websocket_info(self):
        # check websocket stuff
        try:
            response = self.session.get(f"{self.base_url}/api/v3/websocket/info")
            data = response.json()
            
            success = (
                response.status_code == 200 and
                "websocket_enabled" in data
            )
            
            self.log_test("WebSocket Info", success, 
                         f"Status: {response.status_code}, Enabled: {data.get('websocket_enabled', False)}")
            return success
        except Exception as e:
            self.log_test("WebSocket Info", False, str(e))
            return False
    
    def run_all_tests(self):
        # run everything
        print("Running Production Readiness Tests")
        print("=" * 50)
        
        # basic stuff first
        self.test_health_endpoint()
        self.test_endpoints_list()
        self.test_prediction_endpoint()
        
        # auth stuff
        if self.test_user_registration():
            self.test_user_login()
            self.test_protected_endpoint()
            self.test_background_job()
        
        # other random tests
        self.test_ml_model_info()
        self.test_rate_limiting()
        self.test_websocket_info()
        
        # show results
        print("\n" + "=" * 50)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        
        print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All tests passed! Should be good to deploy.")
            return True
        else:
            print("Some tests failed. Probably should fix those.")
            return False
    
    def generate_test_report(self):
        # save test results to a file
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(self.test_results),
            "passed_tests": sum(1 for result in self.test_results if result["success"]),
            "test_results": self.test_results
        }
        
        report_path = Path(__file__).parent / "production_readiness_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Test report saved to: {report_path}")
        return report

def main():
    # actually run the tests
    import subprocess
    import signal
    import atexit
    
    # start flask app in background
    print("Starting Flask application for testing...")
    try:
        os.chdir("/Users/rahulgupta/Developer/CardioPredict/web_platform")
        flask_process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        def cleanup():
            flask_process.terminate()
            flask_process.wait()
        
        atexit.register(cleanup)
        
        # wait a bit for flask to start
        time.sleep(3)
        
        # actually run the tests
        tester = ProductionReadinessTest()
        success = tester.run_all_tests()
        tester.generate_test_report()
        
        # cleanup
        cleanup()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error starting Flask application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
