from astropy.time import Time
from astropy.table import Table, MaskedColumn
import numpy as np


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


def getValue(t, valueT, k1, k2):
    """
    Parameters
    ----------
    t : value of key where we are interested in
    """
    if not np.all(valueT[k1][1:] - valueT[k1][:-1] >= 0):
        raise ValueError("valueT is not monotonic increasing in dates")
    i = np.searchsorted(valueT[k1], t)
    if i < len(valueT):
        if valueT[k1][i] == t:
            return valueT[k2][i]
        if i == 0:
            print("WARNING: Value data starts later than interested dates")
            return valueT[k2][0]
        vlast = valueT[k2][i-1]
        return vlast + (valueT[k2][i] - vlast) * (t - valueT[k1][i-1]) / (valueT[k1][i] - valueT[k1][i-1])
    else:
        print("WARNING: Value data ends earlier than interested dates")
        return valueT[k2][-1]


def prepareRawTransactions(traw):
    """In-place modification of raw loaded csv table"""
    traw['date'] = Time(traw['date'])
    traw.rename_column('date', 'dateS')
    traw.add_column(MaskedColumn(traw['dateS'], name='dateE', mask=True))
    # traw.add_column(np.arange(1, len(traw)+1), name='pId')
    return traw


def packetizeTransactions(traw):
    """Creates a 1-to-1 buy-sell transaction pairing, FIFO order

    Currently supports one asset type only.

    Parameters
    ----------
    traw : `astropy.table.Table`
    'tranType' : 'sell', 'buy'
    'unitPrice' : float
    'quantity' : number of units bought/sold
    """
    packets = traw[traw['type'] == 'buy']
    packets.sort(keys=['dateS', ])
    packets.remove_columns(['type', 'total price'])
    packets.rename_columns(['unit price'], ['priceS'])
    packets.add_column(MaskedColumn(np.zeros(len(packets)), name='priceE', mask=True))
    packets.add_column(np.arange(1, len(packets)+1), name='pId')

    selltable = traw[traw['type'] == 'sell']
    npId = len(packets)+1
    for sale in selltable:
        i_consider = np.searchsorted(packets['dateS'], sale['dateS'], side='right')
        i_unclosed = np.flatnonzero(packets['dateE'].mask[:i_consider])
        saleUnits = sale['quantity']
        ii = 0
        while saleUnits > 0.:
            if len(i_unclosed) <= ii:
                raise ValueError("No match found for sale.")
            irec = i_unclosed[ii]
            packets['dateE'].mask[irec] = False
            packets['dateE'][irec] = sale['dateS']
            packets['priceE'][irec] = sale['unit price']
            if saleUnits > packets['quantity'][irec]:
                saleUnits -= packets['quantity'][irec]  # Still to match sale
            else:
                r = packets['quantity'][irec] - saleUnits
                packets.insert_row(irec+1, packets[irec])
                packets['quantity'][irec] = saleUnits
                irec += 1
                packets['quantity'][irec] = r
                packets['dateE'].mask[irec] = True
                packets['priceE'].mask[irec] = True
                packets['pId'][irec] = npId
                npId += 1
                saleUnits = 0.
            ii += 1
    return packets


def calculateYearMMGain(packets, gainsT=None, asset, tyear, valueT):
    """
    Must be used in incrementing tax years.
    Goes through the packets and calculates either its tax year MM gain,


    gainsT : gains table
    asset : asset id
    tyear : tax year, must be incremental
    totGain : total gain since buy if it was sold this tax year, otherwise 0
    yGain : yearly gain MM or loss if it was not sold this tax year
    urevInc : Unreversed inclusion at the end of the tax year, that
              includes present year.
    """

    flt = packets['asset'] == asset
    D = packets[flt]


    if gainsT is None:
        pass  # Initialize empty table
    yGain = 0
    flt = (gainsT['asset'] == asset) & (gainsT['year'] == tyear - 1)
    flt = np.flatnonzero(flt)
    if len(flt) == 0:
        print(f"First year {tyear} for {asset}.")
        urevInc = 0  # Init from table
    elif len(flt) > 1:
        raise ValueError(f"Multiple entries for {tyear-1} {asset}")
    else:
        urevInc = gainsT['urevInc'][flt]

    totGain = 0
    yearS = Time(f"{tyear:d}-1-1")
    yearE = Time(f"{tyear+1:d}-1-1")

    for pkt in D:
        if pkt['dateS'] >= yearE or (not pkt['dateE'].mask and pkt['dateE'] < yearS):
            # We don't hold this packet during this tax year
            continue

        # This packet's contribution in this year
        dyGain = 0
        dUrevInc = 0
        dTotGain = 0

        if pkt['dateE'].mask:
            if pkt['dateS'] < yearS:
                overlapS = yearS
            else:
                overlapS = pkt['dateS']
            delta = getValue(yearE, valueT, 'date', 'price') \
                - getValue(overlapS, valueT, 'date', 'price')
            delta *= pkt['quantity']
            if delta > 0:
                dyGain = delta
                dUrevInc = delta
            elif delta < 0 and urevInc > 0:
                d2 = min(abs(delta), urevInc)
                dUrevInc = -d2
                dyGain = -d2
        else:
            # This packet was sold during this tax year
            delta = getValue(pkt['dateE'], valueT, 'date', 'price') \
                - getValue(pkt['dateS'], valueT, 'date', 'price')
            delta *= pkt['quantity']
            if delta < 0 and urevInc > 0:
                d2 = min(abs(delta), urevInc)
                dUrevInc = -d2
                delta -= d2
            dTotGain = delta
        totGain += dTotGain
        urevInc += dUrevInc
        yGain += dyGain
        print(f"{tyear:4d} {pkt['pId']} : {dTotGain:6.0f} {dyGain:6.0f} {dUrevInc:6.0f} | "
              f"{totGain:6.0f} {yGain:6.0f} {urevInc:6.0f}")
    gainsT.add_row([asset, tyear, totGain, yGain, urevInc])

    return gainsT
