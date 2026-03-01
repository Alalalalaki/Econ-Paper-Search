#!/bin/bash

# Pull in Econ-Paper-Scrape
echo "Pulling latest paper data..."
cd ../Econ-Paper-Scrape/
git pull

# Copy only CSVs that actually changed (preserves mtime for unchanged files)
echo "Copying updated paper data..."
for f in b2000 2000s 2010s 2015s 2020s 2025s; do
    if ! diff -q "Data/papers_${f}.csv" "../Econ-Paper-Search/Data/papers_${f}.csv" > /dev/null 2>&1; then
        cp "Data/papers_${f}.csv" "../Econ-Paper-Search/Data/"
        echo "  Updated papers_${f}.csv"
    fi
done

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
