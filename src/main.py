from src.broker.ibkr import connect_ibkr

def main():
    ib = connect_ibkr(port=7497, client_id=1)
    acc = ib.accountSummary()
    # petit print propre
    equity = [x.value for x in acc if x.tag == "NetLiquidation"]
    print("✅ Connected to IBKR Paper")
    print("NetLiquidation:", equity[0] if equity else "N/A")
    ib.disconnect()

if __name__ == "__main__":
    main()
