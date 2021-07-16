from astropy.tables import Table


class AssetHistory:
    def ingestTransactions(self, T):
        """Creates a 1-to-1 buy-sell transaction pairing, FIFO order

        Parameters
        ----------
        T : `astropy.table.Table`
        'assetId' : unique id
        'tranType' : 'sell', 'buy'
        'unitPrice' : float
        'quantity' : number of units bought/sold
        """
        pass


class ValueTable:
    """Stores known valuation points then interpolates if necessary for retrieval."""
    pass
