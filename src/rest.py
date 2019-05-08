#!/usr/bin/env python3
from flask import Flask
from flask_restful import Api, Resource, reqparse
import fire
import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder

app = Flask(__name__)
api = Api(app)


#!/usr/bin/env python3


def interact_model(
    model_name='mixed',
    seed=None,
    nsamples=1,
    batch_size=1,
    #length=None,
    length=20,
    temperature=1,
    top_k=0,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        class Sample(Resource):
            def get(self, name):
                #raw_text = input("Model prompt >>> ")
                raw_text = "here's some text" 
                context_tokens = enc.encode(raw_text)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        print(text)
                print("=" * 80)
                return(text)

            def post(self, name):
                parser = reqparse.RequestParser()
                parser.add_argument("prompt")
                args = parser.parse_args()
                samples = []
                context_tokens = enc.encode(args["prompt"])
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
#                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        sample = {
                            "name": "handey",
                            "prompt": args["prompt"],
                            "text": text
                        }
                        samples.append(sample)
                return samples, 201


        api.add_resource(Sample, "/sample/<string:name>")
        app.run(host='0.0.0.0', port=8000)

if __name__ == '__main__':
    fire.Fire(interact_model)

