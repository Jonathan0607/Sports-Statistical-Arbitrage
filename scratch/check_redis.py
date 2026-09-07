import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
state = r.json().get('board:state:kalshi')

brunson_props = [p for p in state.values() if 'brunson' in p.get('player', '').lower()]
for prop in brunson_props:
    print(prop)
