from ib_insync import IB

def connect_ibkr(host="127.0.0.1", port=7497, client_id=1) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib

