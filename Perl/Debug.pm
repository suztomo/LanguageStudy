package Debug;
use strict;
use warnings;

sub new {
  my $pkg = shift;
  my $instance = {
    'version' => 0.1,
    'oldout' => undef,
    'olderr' => undef
  };
  
open($instance->{'oldout'}, ">&STDOUT");
open($instance->{'olderr'}, ">&", \*STDERR);
open(STDOUT, '>', "foo.out");
open(STDERR, ">&STDOUT");
select STDERR; $| = 1;
select STDOUT; $| = 1;
print "Hello, World from Debug.pm";

  bless $instance, $pkg;
}

sub stop {
  my $self = shift;
#  close STDOUT;
  open(STDOUT, ">&", $self->{oldout});
  open(STDERR, ">&", $self->{olderr});
  # stop
}
1;
