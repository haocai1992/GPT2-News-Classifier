#!/bin/sh

echo "----------Downloading model from Google Drive-------------"
wget --no-check-certificate 'https://drive.google.com/uc?export=download&confirm=WFJn&id=1mVWpKZyqM6hbdV0E8REl9m2xuywecs6f' -O gpt2-text-classifier-model.pt
