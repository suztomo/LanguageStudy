{-# OPTIONS_GHC -fno-warn-orphans #-}
{-# LANGUAGE QuasiQuotes, TemplateHaskell, MultiParamTypeClasses,
    OverloadedStrings, TypeFamilies, GADTs, FlexibleContexts, CPP #-}

-- Command:
--   runhaskell runWarp.hs Development

import Yesod.Default.Main
import Yesod.Default.Config
import Application

main :: IO ()
main = defaultMain fromArgs withBulletinBoard