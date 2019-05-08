#!/bin/sh
echo "Fetching data"
curl https://storage.googleapis.com/handey/data.tgz | tar xvz data/
echo "Fetching models"
curl https://storage.googleapis.com/handey/models.tgz | tar xvz models/
