{-
$ ghc --make 3.8_Interactivity.hs -o test
$ ./test
-}
module Main
    where

import IO
main = do
  hSetBuffering stdin LineBuffering
  putStrLn "Please enter your name: "
  name <- getLine
  putStrLn("Hello, " ++ name ++ ", how are you?")

