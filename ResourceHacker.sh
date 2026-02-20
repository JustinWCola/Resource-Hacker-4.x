#!/bin/sh
rm ./tec/*
mpirun -np 8 python main.py

