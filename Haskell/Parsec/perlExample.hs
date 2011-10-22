import Text.ParserCombinators.Parsec
import Text.ParserCombinators.Parsec.Expr
import Text.ParserCombinators.Parsec.Language
import qualified Text.ParserCombinators.Parsec.Token as P

data Identifier = Identifier String deriving (Show, Read, Eq)

data FuncDef = FuncDef {
      name :: Identifier,
      body :: String } deriving Show



perlFile :: GenParser Char st [ FuncDef ]
perlFile =
    do result <- many funcDef
       eof
       return result

languageDefinition = javaStyle {
                       reservedNames = ["sub"]
                     }

lexer = P.makeTokenParser languageDefinition

reserved              = P.reserved lexer
symbol                = P.symbol lexer
identifierLexer       = P.identifier lexer  
whiteSpace            = P.whiteSpace lexer

identifier = do {
  x <- identifierLexer;
  return(Identifier x)
}

funcDef :: GenParser Char st FuncDef
funcDef =
    do
      reserved "sub"
      name <- identifierLexer
      whiteSpace
      symbol "{"
      body <- funcBody
      symbol "}"
      return (FuncDef (Identifier name) body)

funcBody :: GenParser Char st String
funcBody =
    do many (noneOf "\n{}")

parsePerlFile :: String -> Either ParseError [ FuncDef ]
parsePerlFile input = parse perlFile "(parse error)" input

t :: Either ParseError [ FuncDef ]
t = parsePerlFile "sub tomeFunc { aiueo } \n\n sub takaFunc { kakikukeko } "
