{-
$ ghc --make 3.8_Interactivity.hs -o test
$ ./test
-}
module Main
    where

import IO
import Random

main = do
  hSetBuffering stdin LineBuffering
  num <- randomRIO (1::Int, 100)
  putStrLn "I'm thinking of a number between 1 and 100"
  doGuessing num

{- ???? -}
{-
doGuessing num =
    do
      if 1 > 0
      then putStrLn "Hello"
      else putStrLn "tako"
-}
signum :: (Num a, Ord a) => a -> a
{-
The statement before '=>' means `a' is an instance of Num:
Int, Double, etc.
signum :: Int -> Int
 -}
signum x =
    if x < 0
      then -1
      else if x > 0
             then 1
             else 0

doGuessing num = do
  putStrLn "Enter your guess: "
  guess <- getLine
  let guessNum = read guess
  if guessNum < num
    then do putStrLn "Too low!"
            doGuessing num
    else if read guess > num
           then do putStrLn "Too high!"
                   doGuessing num
           else do putStrLn "You Win!"



