import streamlit as st


def faq():
    st.markdown(
        """
# FAQ
## Was genau ist mit den AWMF Leitlinien gemeint?
Die Datenbank besteht aus den deutschen medizinischen [aktuellen Leitlinien](https://register.awmf.org/de/leitlinien/aktuelle-leitlinien), einsehbar auf AWMF online.
Dies umfasst 931 Dokumente mit gesamt 86.922 Seiten.
An diese Datenbank wird eine Suchmaschine angeschlossen, 
um Fragen mit den best passensten Antworten in den Leitlinien verbinden zü können.
Die Datenbank Option "Nur aktuell gültige Leitlinien" besteht nur aus den noch nicht abgelaufenen Leitlinien.

## Sind meine Daten sicher?
Diese App speichert keine Nutzerbezogenen Daten und hält auch 
den Chatverlauf nicht länger als notwendig für deinen Besuch auf dieser Seite.

## Wie genau wird die Antwort gegeben?
Die 60.000 Seiten der Leitlinien werden in kleinere Abschnitte unterteilt 
und in einem speziellen Datenbanktyp namens Vektorstore 
gespeichert, der eine semantische Suche und Abfrage ermöglicht.
Wenn Sie eine Frage stellen, durchsucht LeitlinienGPT die
Dokumententeile und findet die relevantesten.
Dies basiert auf dem Grad der ähnlichkeit der Vektoren von 
Frage und allen Abschnitten im Vektorstore.
Dann wird OpenAI's GPT-3.5 Modell verwendet, um eine endgültige Antwort zu generieren.

## Was bedeuten die Informationen unter jeder Quelle?
z. B. [Adipositas - 050-002l_S3_Therapie-Praevention-Adipositas-Kinder-Jugendliche_2019-11]
Unter jeder Quelle wird zuerst die Kategorie gelistet unter
welcher die Leitlinie im AWMF Register zu finden ist.
Danach folgt der Name der Leitlinie selbst.
Als Letztes wird die Seitenzahl des Dokumentes
aufgeführt, in welchem die Antwort zu finden ist.

## Sind die Antworten 100 % genau?
Nein, die Antworten sind nicht 100%ig genau. LeitlinienGPT verwendet GPT-3.5 zur Generierung
von Antworten. GPT-3.5 ist ein leistungsfähiges Sprachmodell, aber es macht manchmal Fehler 
und ist anfällig für Halluzinationen. Außerdem verwendet LeitlinienGPT eine semantische Suche
um die relevantesten Abschnitte zu finden und sieht nicht das gesamte Dokument,
was bedeutet, dass es möglicherweise nicht alle relevanten Informationen findet und
nicht alle Fragen beantworten kann (vor allem Fragen vom Typ Zusammenfassung
oder Fragen, die eine Menge Kontext aus dem Dokument erfordern).

Aber für die meisten Anwendungsfälle ist KnowledgeGPT sehr genau und kann die
die meisten Fragen beantworten. Überprüfen Sie immer die Quellen, um sicherzustellen, 
dass die Antworten korrekt sind.
"""
    )