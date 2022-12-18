from flask import Flask,jsonify
from flask_restful import Resource, Api
import numpy as np
import pymongo
# importing ObjectId from bson library
from bson.objectid import ObjectId
import json
from pyeditorjs import EditorJsParser

import settings
from utils import editorJsDataToText

client = pymongo.MongoClient(settings.MONGODB_SETTINGS["host"])
db = client[settings.MONGODB_SETTINGS["db"]]
mongo_col = db[settings.MONGODB_SETTINGS["collection"]]

app = Flask(__name__)
api = Api(app)

class Post(Resource):
    def get(self, post_id):
        main_post = mongo_col.find_one({"_id": ObjectId(post_id)})
        main_post= {
            "_id": str(main_post["_id"]),
            "text": editorJsDataToText(json.loads(main_post["content"])),
            "title": main_post["title"],
            "subtitle": main_post["subtitle"],
            "content": main_post["content"]

        }

        return jsonify({"post": main_post})

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')
api.add_resource(Post, '/posts/<post_id>')

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)