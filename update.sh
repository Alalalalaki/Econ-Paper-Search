#! /bin/sh.

# pull in Econ-Paper-Scrape
cd ../Econ-Paper-Scrape/
git pull

# move data file
cp Data/papers_2020s.csv ../Econ-Paper-Search/Data/

# commit and push in Econ-Paper-Search
cd ../Econ-Paper-Search
git ci "Monthly data update."
git push
