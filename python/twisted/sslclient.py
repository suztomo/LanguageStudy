from twisted.internet import ssl, reactor
from twisted.internet.protocol import ClientFactory, Protocol, ClientCreator
from twisted.words.protocols.irc import IRCClient

class Mikan(IRCClient):
	username = "mikan"
	userinfo = "mikan IRC Bot"
	versionName = "0.1"
	sourceURL = "http://github.com/suztomo/"
	channel = "#suztomo"
	nickname_ = "mikan"

	def _get_nickname(self):
		return self.nickname_
	nickname = property(_get_nickname)

	def signedOn(self):
		self.join(self.channel)
		print "Signed on as %s." % (self.nickname,)

	def joined(self, channel):
		print "Joined %s." % (channel,)

	def privmsg(self, user, channel, msg):
		print msg

class IRCClientFactory(ClientFactory):
	protocol = Mikan

	def clientConnectionFailed(self, connector, reason):
		print "Connection failed - goodbye!"
		reactor.stop()

	def clientConnectionLost(self, connector, reason):
		print "Connection lost - goodbye!"
		reactor.stop()

from threading import Thread
from time import sleep

class twitterThread(Thread):
	def __init__(self, rct):
		Thread.__init__(self)
		self.ircreactor = rct
	def run(self):
		count = 0
		while count < 10:
			print "twitterThread"
			sleep(1)
			count += 1
			ircreactor.callFromThread(


if __name__ == '__main__':
	twt = twitterThread(reactor)
	twt.start()
	factory = IRCClientFactory()
	reactor.connectSSL('cs2009.org', 6666, factory, ssl.ClientContextFactory())
	reactor.run()


