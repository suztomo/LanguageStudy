package Another;
use feature qw( say );

use strict;
use warnings;

sub new {
  my $pkg = shift;
  bless {}, $pkg;
}

sub hello {
  say "Hello, World from another";
}

1;
