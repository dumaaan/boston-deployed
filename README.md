# Deploying an ML system in production project

In this project, I am trying to learn more anout using Docker, Flask, and how to deploy a small ML model into production for others.

**Command to run a Docker container:**

`docker run -it -p 5000:5000 docker-api python3 api.py`

**Sending a prediction request:**
```
curl -i -H "Content-Type: application/json" -X POST \
	-d '{"CRIM": YOURVALUE, "ZN": YOURVALUE, "INDUS": YOURVALUE, "CHAS": YOURVALUE, "NOX": YOURVALUE, \
		 "RM": YOURVALUE, "AGE": YOURVALUE, "DIS": YOURVALUE, "RAD": YOURVALUE, "TAX": YOURVALUE, \
		 "PTRATIO": YOURVALUE, "B": YOURVALUE, "LSTAT": YOURVALUE}' \
	    127.0.0.1:5000/predict
```

The project is far from being finished so I will update it along the way.


**TO-DO:**
* Creating a web interface instead of sending a request through the terminal
* Fine-tuning a model
