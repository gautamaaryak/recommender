# app.py
from flask import Flask, request, jsonify
import model
app = Flask(__name__)

@app.route('/getrec/', methods=['GET'])
def respond():
    # Retrieve the name from url parameter
    name = request.args.get("name", None)

    # For debugging
    print(f"got name {name}")

    response = {}

    # Check if user sent a name at all
    if not name:
        response["ERROR"] = "no name found, please send a name."
    # Check if the user entered a number not a name
    elif str(name).isdigit():
        response["ERROR"] = "name can't be numeric."
    # Now the user entered a valid name
    else:
        response["MESSAGE"] = model.get_products(name)

        # Return the response in json format
        line1 =  "<h1>Here is the recommendation for "+ name + "<h1> <br> <ul>"

        for i in response["MESSAGE"]:
            line1 = line1 + "<li>" + i + "</li>"

        line1 = line1 + "</ul>"
        response = line1

    return response


# A welcome message to test our server
@app.route('/')
def index():
    with open('index.html', 'r') as f:
        data = f.read()
    # return """<h1>Welcome to our server !!</h1><br> Please input name <form action="getrec" method="get"> <input type="text" name="name">  <input type="submit" name="submit"></form>"""
    return data

    
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)