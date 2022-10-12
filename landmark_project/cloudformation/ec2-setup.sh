#!/bin/bash
sudo apt-get update -y
sudo apt install python3-pip -y
sudo apt install unzip -y
sudo apt install git -y
sudo pip install numpy pandas matplotlib jupyter pyyaml joblib scikit-learn scipy opencv-python
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
sudo rm awscliv2.zip
cd ~
aws s3 cp s3://uj-scratch/udacity/ec2-setup.sh .
chmod +x ec2-setup.sh
mkdir udacity-project2
cd udacity-project2
sudo curl "https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip" -o "landmark_images.zip"
sudo unzip landmark_images.zip
sudo rm landmark_images.zip
aws s3 cp s3://uj-scratch/udacity/image_data_utils.py .
aws s3 cp s3://uj-scratch/udacity/landmark.ipynb .
aws s3 cp s3://uj-scratch/udacity/torch_simulate.py .
mkdir references
cd references
aws s3 sync s3://uj-scratch/udacity/references .
cd ..
mkdir images
aws s3 sync s3://uj-scratch/udacity/images .
rm landmark_images.zip
ls