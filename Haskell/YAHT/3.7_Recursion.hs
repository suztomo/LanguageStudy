module Main
    where

factorial 1 = 1
factorial n = n * factorial (n-1)

exponent a 1 = a
exponent a n = a * (Main.exponent a (n-1))

