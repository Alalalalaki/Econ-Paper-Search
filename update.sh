#!/bin/bash

# Pull in Econ-Paper-Scrape
echo "Pulling latest paper data..."
cd ../Econ-Paper-Scrape/
git pull

# Move data file
echo "Copying updated paper data..."
cp Data/papers_2020s.csv ../Econ-Paper-Search/Data/

# Update embeddings for modified files
echo "Updating embeddings for modified data..."
cd ../Econ-Paper-Search/Code
python update_embedding.py
python test_embedding_consistency.py

# Go back to repo root
cd ..

# Commit and push in Econ-Paper-Search
echo "Committing and pushing updates..."
git add -A
git commit -m "Monthly update"
git push

echo "Update complete!"
