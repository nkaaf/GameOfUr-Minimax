Für Version1: Würfeln vor der Minimax Berechnung -> Erhebliche Verringerung der Zustände

Ideen für Nutzenfunktion:
- Spielsteine außerhalb von Range des Gegners bezogen auf den nächsten Zug vom Gegner (gut)
- Bevorzugt spielsteine auf Spezialfelder/Safe-Spots stellen
- Spielsteine auf eigene Safe-Zone packen
- Felder mit Punkten versehen, um die Bedeutung eines jeden Feldes zu gewichten
- Immer im Hinterkopf behalten, dass eine 2 zu würfeln am wahrscheinlichsten ist. (Beispiel: Stein steht auf der 0 und auf der 9. Besser die 0 -> 2 als 9 -> 11, da der Zug von 2-> 4 besser ist um nochmal zu würfeln).
- Nächster Zug vom Gegner berücksichtigen (2 am Wahrscheinlichsten)




[0,0,2,0,0,0,0,2,0,0,0,0,0,0,1,0,0,0,1,0]

0 -> niemand drauf
1 -> Spieler 1
2 -> Spieler 2
3 -> Spezialfeld

order1: [4,3,2,1,0,5,6,7,8,9]
places1: [12,16,-1,-2,-1,-1,-1]
-1 -> Stein ist am Start
-2 -> Stein ist im Ziel
score1 = count(-2 in places1)
score2 = count(-2 in places2)