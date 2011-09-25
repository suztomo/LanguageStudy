http://www.openldap.org/doc/admin24/quickstart.html


We have to change "rootpw" according to 
http://www.atmarkit.co.jp/flinux/rensai/openldap04/openldap04a.html


When I tried to add example.ldif I encountered error that 
  ~/Documents/LanguageStudy/LDAP $ ldapadd -x -D "cn=Manager,dc=suztomo,dc=net" -w secret -f example.ldif 
  adding new entry "dc=suztomo,dc=net"
  ldap_add: Invalid syntax (21)
	additional info: objectclass: value #0 invalid per syntax

This was caused by the space charactor trailing objectclass lines.

Password
slappasswd -h {SSHA} -s secret

Search
ldapsearch -x -b 'dc=suztomo,dc=net' '(objectclass=*)'
Search doesn't require password

Add
ldapadd -x -D "cn=Manager,dc=suztomo,dc=net" -w secret -f ex.ldif
The password corresponds to the line "rootpw" in slapd.conf.


Adding error.
~/Documents/LanguageStudy/LDAP $ cat director.ldif 
dn: cn=Director,dc=suztomo,dc=net
objectclass: organizationalRole
cn: Directer

This causes...

~/Documents/LanguageStudy/LDAP $ ldapadd -x -D "cn=Manager,dc=suztomo,dc=net" -w secret -f director.ldif 
adding new entry "cn=Director,dc=suztomo,dc=net"
ldap_add: Naming violation (64)
	additional info: value of naming attribute 'cn' is not present in entry

Run
sudo /usr/libexec/slapd -f /opt/local/etc/openldap/slapd.conf -d -1 

~/Documents/LanguageStudy/LDAP $ netstat -an |grep 389
tcp6       0      0  *.389                  *.*                    LISTEN
tcp4       0      0  *.389                  *.*                    LISTEN


When 
