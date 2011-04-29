package net.suztomo.ponta

import scala.util.parsing.json.JSON._
import scala.io.Source

object configReader {
  def read(filename:String):Any = {
    val lines = Source.fromFile(filename).mkString
    parseFull(lines) match {
      case Some(map) => {
        return map
      }
      case _ => {
        println("invalid json")
        return null
      }
    }
  }
}
