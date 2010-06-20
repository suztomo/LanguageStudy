module Main
    where

data Configuration =
    Configuration { username :: String,
                    localhost :: String,
                    remotehost :: String,
                    isGuest :: Bool,
                    timeconnected :: Integer
                  }

instance Show Configuration where
    show c = "Configuration(" ++ (username c) ++ ", " ++ (remotehost c) ++ ")"

initCFG = Configuration "nobody" "nowhere" "nowhere" False 0

getHOstData (Configuration {localhost = lh, remotehost = rh}) = (lh, rh)


data Tree a
    = Leaf a
    | Branch (Tree a) (Tree a)

-- :kind FuncInt
-- TupleThree :: * -> * -> * -> *
data FuncInt a b c = FuncInt(a -> b -> c)

-- :kind TupleThree
-- TupleThree :: * -> * -> * -> *
data TupleThree a b c = TupleThree(a, b, c)

a = Leaf 'c'


{-
  "Eq a" is a context
  ConsSet can only be applied to values whose type is an instance of the class Eq.

  Pattern matching against ConsSet also gives rise to an Eq a constraint
  for example,
    f (ConsSet a s) = a
  the function f has inferred type Eq a => Set a -> a. The context in the data declaration
  has no other effect whatsoever.
-}
data Eq a => Set a = NilSet | ConsSet a (Set a)


data C = F { f1, f2 :: Int, f3 :: Bool}
         deriving (Eq, Ord, Show)         

{- defines a type and constructor identical to the one produced by

data C = F Int Int Bool
-}


