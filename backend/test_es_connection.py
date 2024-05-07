from elasticsearch import Elasticsearch

# Ersetzen Sie diese Werte durch Ihre tatsächlichen Elasticsearch-Zugangsdaten
es_cloud_id="LeitlinienGPT:ZXUtY2VudHJhbC0xLmF3cy5jbG91ZC5lcy5pbzo0NDMkYjQ2M2U4MmFhMTU3NDk0MWE2YTZlMjkxNzRmY2FjYjYkNzJlMDU1NzEyZjM5NDU3NTgxNTUyZDFlODFiMDE0YmY="
es_user="enterprise_search"
es_password="-VwsG8mt-TELfRQ"

# Verbindung zu Elasticsearch herstellen
es = Elasticsearch(
    cloud_id=es_cloud_id,
    http_auth=(es_user, es_password)
)

# Führen Sie eine kleine Suche aus, um die Verbindung zu testen
response = es.info()

# Ausgabe der Antwort
print(response)
