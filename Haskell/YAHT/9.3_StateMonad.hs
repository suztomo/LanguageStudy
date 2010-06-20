module Main
    where

data Tree a
    = Leaf a
    | Branch (Tree a) (Tree a)

instance Show Tree  where
    show (Leaf a) = "Leaf " ++ (show a)
    show (Branch a b) = "Branch (" ++ (show a) ++ ", " ++ (show b) ++ ")"

{-
mapTree :: (a -> b) -> Tree a -> Tree b
mapTree f (Leaf a) = Leaf (f a)
mapTree f (Branch lhs rhs) =
    Branch (mapTree f lhs) (Maptree f rhs)

mapTreeState :: (a -> state -> (state, b) ->
                 Tree a -> state -> (state, Tree b))
mapTreeState f (Leaf a) state =
    let (state', b) = f a state
    in (state', Leaf b)

mapTreeState f (Branch lhs rhs) state =
    let (state' , lhs') = mapTreeState f lhs state
        (state'', rhs') = mapTreeState f rhs state'
    in (state'', Branch lhs' rhs')
-}

{-
type State st a = st -> (st, a)

returnState :: a -> State st a
return State a = \st -> (st, a)

bindState :: State st a -> (a -> State st b) ->
             State st b
-}
{-
  m : (State st a), an action that returns something of type `a' with state `st'
      (st -> (st, a))
  k : (a -> State st b), a function that takes this `a' and produces something of type `b'
    : (a -> st -> (st, b))

  (m' st') :: (st, a) . in addition to this, "\st ->" makes bindState "st -> (st, a)"
-}

{-
bindState m k = \st ->
                let (st', a) = m st
                    m'       = k a
                in m' st'

mapTreeStateM :: (a -> State st b) -> Tree a -> State st (Tree b)
mapTreeStateM f (Leaf a) =
    f a `bindState` \b ->
    returnState (Leaf b)
mapTreeStateM f (Branc Lhs rhs) =
    mapTreeStateM f lhs `bindState` \lhs' ->
    mapTreeStateM f rhs `bindState` \rhs' ->
    returnState (Branch lhs' rhs')
-}

newtype State st a = State (st -> (st, a))

instance Monad (State state) where
    return a = State (\state -> (state, a))
    State run >>= action = State run'
        where run' st =
                  let (st', a) = run st
                      State run'' = action a
                  in run'' st'

mapTreeM :: (a -> State state b) -> Tree a -> State state (Tree b)
mapTreeM f (Leaf a) = do
  b <- f a
  return (Leaf b)

mapTreeM f (Branch lhs rhs) = do
  lhs' <- mapTreeM f lhs
  rhs' <- mapTreeM f rhs
  return (Branch lhs' rhs')

getState :: State state state
getState = State (\state -> (state, state))

putState :: state -> State state ()
putState new = State (\_ -> (new, ()))

numberTree :: Tree a -> State Int (Tree (a, Int))
numberTree tree = mapTreeM number tree
    where number v = do
            cur <- getState
            putState (cur + 1)
            return (v, cur)

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

runStateM :: State state a -> state -> a
runStateM (State f) st = snd (f st)


