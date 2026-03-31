wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz -O images.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz -O annotations.tar.gz
mkdir -p dataset
tar -xzf images.tar.gz -C dataset
tar -xzf annotations.tar.gz -C dataset
rm annotations.tar.gz images.tar.gz