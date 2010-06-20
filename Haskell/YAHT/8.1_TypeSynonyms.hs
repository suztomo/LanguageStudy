module Main
    where

type List3D = [(Double, Double, Double)]

p = (3.2, 3.3, 1.1)


getFirst :: List3D -> Double
getFirst ((a, _, _):_) = a
getFirst _ = 0.0
{-
Ok, modules loaded: Main.
*Main> p
(3.2,3.3,1.1)
*Main> [p]
[(3.2,3.3,1.1)]
*Main> getFirst [p]
3.2
-}

newtype MyInt = MyInt Int

instance Show MyInt where
    show (MyInt a) = "MyInt " ++ (show a)

data Tree a
    = Leaf a
    | Branch (Tree a) (Tree a)

showTree                :: (Show a) => Tree a -> String
showTree (Leaf x)       =  "Leaf " ++ (show x)
showTree (Branch l r)   =  "<" ++ showTree l ++ "|" ++ showTree r ++ ">"

{- http://www.haskell.org/tutorial/stdclasses.html -}

instance Show a => Show (Tree a)  where
    show = showTree

instance Show (Tree Int) where
    show = showTree
{-
8.1_TypeSynonyms.hs:40:14:
    `Tree' is not applied to enough type arguments
    The first argument of `Show' should have kind `*',
    but `Tree' has kind `* -> *'
    In the instance declaration for `Show Tree'

The type constructor Tree has the kind *->*; the type Tree Int has the kind *.
http://www.haskell.org/tutorial/classes.html
-}

testTree =
    Branch
      (Branch
        (Leaf 'a')
        (Branch
          (Leaf 'b')
          (Leaf 'c')))
      (Branch
        (Leaf 'd')
        (Leaf 'e'))


