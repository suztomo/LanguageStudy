{-# LANGUAGE TypeFamilies, QuasiQuotes, OverloadedStrings, MultiParamTypeClasses, TemplateHaskell #-}
import Yesod
import Control.Applicative
import Data.Text (Text)

data FormExample = FormExample
-- type Handler = GHandler FormExample FormExample

mkYesod "FormExample" [$parseRoutes|
/ RootR GET
|]

instance Yesod FormExample where
    approot _ = ""

data Person = Person { name :: Text, age :: Int }
    deriving Show

personFormlet p = fieldsToTable $ Person
    <$> stringField "Name" (fmap name p)
    <*> intField "Age" (fmap age p)

getRootR :: Handler RepHtml
getRootR = do
    (res, wform, enctype) <- runFormGet $ personFormlet Nothing
    defaultLayout $ do
        setTitle "Form Example"
        form <- extractBody wform
        addHamlet [$hamlet|
<p>Last result: #{show res}
<form enctype="#{enctype}">
    <table>
        \^{form}
        <tr>
            <td colspan="2">
                <input type="submit">
|]

main = warpDebug 3000 FormExample