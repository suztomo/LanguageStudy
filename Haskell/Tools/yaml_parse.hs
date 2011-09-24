{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

import System.IO  
import Text.Printf
import System.Environment (getArgs)
import Data.Object.Yaml
import Data.Object
import Control.Monad
import Control.Failure
import Control.Applicative
import List

getPeopleName :: (Failure ObjectExtractError m)
                 => [(String, Object String String)] -> m String
getPeopleName p = lookupScalar "name" p

peopleNameList :: (Failure ObjectExtractError m)
                  => [[(String, Object String String)]] -> m [String]
peopleNameList people =
    mapM getPeopleName people
--    let a = map getPeopleName people
--    in sequence a
--      return ["aiueo", "kaki"] :: (Failure ObjectExtractError m) => m [String]

fmp :: (Failure ObjectExtractError m) =>
       Object String String -> m [(String, Object String String)]
fmp = fromMapping

fm :: (Failure ObjectExtractError m) =>
      [Object String String] -> [m [(String, Object String String)]]
fm people = map fmp people

convertHTMLList :: [String] -> String
convertHTMLList lst = "<ul><li>"
                      ++ (concat $ intersperse "</li><li>" lst)
                      ++ "</li></ul>"

processYamlFile :: String -> IO ()
processYamlFile fileName =
    do
      object <- join $ decodeFile fileName :: IO (Object String String)
      people <- fromSequence object
             :: (Failure ObjectExtractError m)
                => m [Object String String]
      let t = fm people
            :: (Failure ObjectExtractError m) =>
               [m [(String, Object String String)]]
      ppl <- sequence t
--            :: (Failure ObjectExtractError m) =>
--               m [[(String, Object String String)]]
      names  <- peopleNameList ppl
             :: (Failure ObjectExtractError m) => m [String]
      let nameInHTML = convertHTMLList names
      putStrLn $ nameInHTML

main = do
  args <- getArgs
  case args of
    [input] -> processYamlFile input
    _ -> fail "one argument is required"
