from flask import Flask, jsonify
from flask_restful import Resource, Api
import numpy as np
import pymongo
from bson.objectid import ObjectId
import json

import settings
from src.distances import get_most_similar_documents
from src.models import make_texts_corpus
from utils import editorJs_data_to_text

client = pymongo.MongoClient(settings.MONGODB_SETTINGS["host"])
db = client[settings.MONGODB_SETTINGS["db"]]
mongo_col = db[settings.MONGODB_SETTINGS["collection"]]

app = Flask(__name__)
api = Api(app)


def load_model():
    import gensim  # noqa
    import joblib  # noqa
    # load LDA model
    lda_model = gensim.models.LdaModel.load(
        settings.PATH_LDA_MODEL
    )
    # load corpus
    corpus = gensim.corpora.MmCorpus(
        settings.PATH_CORPUS
    )
    # load dictionary
    id2word = gensim.corpora.Dictionary.load(
        settings.PATH_DICTIONARY
    )
    # load documents topic distribution matrix
    doc_topic_dist = joblib.load(
        settings.PATH_DOC_TOPIC_DIST
    )
    # doc_topic_dist = np.array([np.array(dist) for dist in doc_topic_dist])

    return lda_model, corpus, id2word, doc_topic_dist


lda_model, corpus, id2word, doc_topic_dist = load_model()


class Post(Resource):
    def get(self, post_id):
        main_post = mongo_col.find_one({"_id": ObjectId(post_id)})
        main_post = {
            "_id": str(main_post["_id"]),
            "text": editorJs_data_to_text(json.loads(main_post["content"])),
            "title": main_post["title"],
            "subtitle": main_post["subtitle"],
            "content": main_post["content"]
        }

        # preprocessing
        content = editorJs_data_to_text(json.loads(main_post["content"]))
        text_corpus = make_texts_corpus([content])
        bow = id2word.doc2bow(next(text_corpus))
        doc_distribution = np.array(
            [doc_top[1] for doc_top in lda_model.get_document_topics(bow=bow)]
        )

        # recommender posts
        most_sim_ids = list(get_most_similar_documents(
            doc_distribution, doc_topic_dist))[1:]

        most_sim_ids = [int(id_) for id_ in most_sim_ids]
        posts = mongo_col.find({"idrs": {"$in": most_sim_ids}})
        related_posts = [
                            {
                                "_id": str(post["_id"]),
                                "title": post["title"],
                            }
                            for post in posts
                        ][1:]

        return jsonify({"post": {
            "_id": str(main_post["_id"]),
            "text": editorJs_data_to_text(json.loads(main_post["content"])),
            "title": main_post["title"],
            "subtitle": main_post["subtitle"]
        },
            "related_posts": related_posts,
        })

class PostsRecommend(Resource):
  def get(self, post_id):
    main_post = mongo_col.find_one({"_id": ObjectId(post_id)}, {"content": 1})
    # preprocessing
    content = editorJs_data_to_text(json.loads(main_post["content"]))
    text_corpus = make_texts_corpus([content])
    bow = id2word.doc2bow(next(text_corpus))
    doc_distribution = np.array(
        [doc_top[1] for doc_top in lda_model.get_document_topics(bow=bow)]
    )
    # recommender posts
    most_sim_ids = list(get_most_similar_documents(
        doc_distribution, doc_topic_dist))[1:]

    most_sim_ids = [int(id_) for id_ in most_sim_ids]
    posts = mongo_col.aggregate([
      {"$lookup": {
        "from": "users",
        "localField": "user.$id",
        "foreignField": "_id",
        "pipeline": [
          {"$project": {"_id": 0, "id": {
            "$toString": "$_id"
          }, "username": 1, "email": 1, "avatar": 1}}
        ],
        "as": "user"
        }},
      {"$match": {"idrs": {"$in": most_sim_ids}}},
      {"$project": {"_id": 0,"id": {
        "$toString": "$_id"
      }, "title": 1, "subtitle": 1,"thumbnail" : 1, "user": {"$arrayElemAt": ["$user", 0]}, "createdDate":{
        "$dateToString": {
          "date": "$createdDate"
        }
      }}},
      {"$sort": {"createdDate": -1}},
      {"$limit": 10}
    ])

    related_posts = [
        {
            "id": post["id"],
            "title": post["title"],
            "user": post["user"],
            "thumbnail": post["thumbnail"] if "thumbnail" in post else None,
            "createdDate": post["createdDate"],
        }
        for post in posts
    ][1:]
    return jsonify(related_posts)




api.add_resource(PostsRecommend, '/api/posts/<post_id>/recommend')
api.add_resource(Post, '/posts/<post_id>')

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
