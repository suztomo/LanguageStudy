use strict;
use warnings;
use feature qw( say );

use Debug;
use Another;
say "before call";
my $d = Debug->new();
my $i = Another->new();
say "Hello, World";
$i->hello();
$d->stop();
say 'After stop';
say $$;
sleep 60;
