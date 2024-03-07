# This is the chatbot for Intellasure
## To chat with DB

```
curl --location 'http://127.0.0.1:5000/chat' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Is LVY prefix out-of-network?"
}'
```