from astropy.time import Time
from astropy.table import Table, MaskedColumn, Column
import numpy as np
import astropy.table


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


def getValue(t, valueT, k_t, k_val):
    """

    Interpolates
    Parameters
    ----------
    t  : value of key where we are interested in
    k_t : key for t in valueT for ``t``
    k_val : key for value in valueT
    """
    if not np.all(valueT[k_t][1:] - valueT[k_t][:-1] >= 0):
        raise ValueError("valueT is not monotonic increasing in dates")
    i = np.searchsorted(valueT[k_t], t)
    if i < len(valueT):
        if valueT[k_t][i] == t:
            return valueT[k_val][i]
        if i == 0:
            print("WARNING: Value data starts later than interested dates")
            return valueT[k_val][0]
        vlast = valueT[k_val][i-1]
        r = (t-valueT[k_t][i-1]) / (valueT[k_t][i]-valueT[k_t][i-1])
        return vlast + (valueT[k_val][i]-vlast) * r.value
    else:
        print("WARNING: Value data ends earlier than interested dates")
        return valueT[k_val][-1]


def getPriceConverted(t, priceT, currencyT=None):
    v = getValue(t, priceT, 'date', 'price')
    if currencyT is not None:
        xchg = getValue(t, currencyT, 'date', 'rate')
    else:
        xchg = 1.
    return v*xchg


def prepareRawTransactions(traw):
    """In-place modification of raw loaded csv table"""
    traw['date'] = Time(traw['date'])
    traw.rename_column('date', 'dateS')
    traw.add_column(MaskedColumn(traw['dateS'], name='dateE', mask=True))
    # traw.add_column(np.arange(1, len(traw)+1), name='pId')
    return traw


def extendPriceTable(priceT, preprawT, assetId):
    """Adds new entries from the packetized transactions to the price table"""
    p2 = preprawT.copy()
    p2.rename_column('dateS', 'date')
    p2.rename_column('unit price', 'price')
    R = astropy.table.vstack([priceT, p2])
    return astropy.table.unique(R, keys=['date', 'price'], keep='first')


def packetizeTransactions(traw, assetPrefix):
    """Creates a 1-to-1 buy-sell transaction pairing, FIFO order

    Currently supports one asset type only.

    Parameters
    ----------
    traw : `astropy.table.Table`
     'tranType' : 'sell', 'buy'
     'unit price' : float
     'quantity' : number of units bought/sold
    assetPrefix : string
      Prefix string for packet ids

    traw will be sorted in-place by date
    """
    traw.sort(keys=['dateS', ])
    packets = traw[traw['type'] == 'buy']
    packets.remove_columns(['type', 'total price'])
    packets.rename_columns(['unit price'], ['priceS'])
    packets.add_column(MaskedColumn(np.zeros(len(packets)), name='priceE', mask=True))
    packets.add_column(Column([f'{assetPrefix}{x:03d}' for x in range(1, len(packets)+1)],
                              dtype='S20'), name='packetId')
    selltable = traw[traw['type'] == 'sell']
    npId = len(packets)+1
    dfmt = "%Y-%m-%d"
    for sale in selltable:
        i_consider = np.searchsorted(packets['dateS'], sale['dateS'], side='right')
        i_unclosed = np.flatnonzero(packets['dateE'].mask[:i_consider])
        L = [f'{x["dateS"].strftime(dfmt)}:{x["quantity"]}@{x["priceS"]}' for x in packets[i_unclosed]]
        print("======")
        print(f"Sale {sale['dateS'].strftime(dfmt)}:{sale['quantity']}@{sale['unit price']} -> \nBuys ")
        print("\n".join(L))
        saleUnits = sale['quantity']
        ii = 0  # Index in i_unclosed
        while saleUnits > 0.:
            print(f"Using buy {ii}")
            if len(i_unclosed) <= ii:
                raise ValueError("Data error - No more buy transaction to match sale.")
            irec = i_unclosed[ii]
            packets['dateE'].mask[irec] = False
            packets['priceE'].mask[irec] = False
            packets['dateE'][irec] = sale['dateS']
            packets['priceE'][irec] = sale['unit price']
            if saleUnits > packets['quantity'][irec]:
                saleUnits -= packets['quantity'][irec]  # Still to match sale
                print(f"Still to sale {saleUnits}")
            else:
                r = packets['quantity'][irec] - saleUnits
                if r > 0.:
                    print(f"Remaining from buy : {r}")
                    packets.insert_row(irec+1, packets[irec])
                packets['quantity'][irec] = saleUnits
                if r > 0.:
                    irec += 1
                    packets['quantity'][irec] = r
                    packets['dateE'].mask[irec] = True
                    packets['priceE'].mask[irec] = True
                    packets['packetId'][irec] = f'{assetPrefix}{npId:03d}'
                    npId += 1
                saleUnits = 0.
            ii += 1
    return packets


def calculateOneYearOneAssetTaxes(packetsT, assetId, tyear, priceT,
                                  packetGainsT=None, totalGainsT=None,
                                  firstMMTaxYear=False):
    """
    Must be used in incrementing tax years.
    Goes through the packets and calculates either its tax year MM gain,

    Parameters
    ==========
    packetsT: `astropy.table.Table`
        Packetized transactions
    assetId : `string`
        asset id
    tyear : `int`
        tax year, must be used in a sequence starting with first MM year
    priceT: `astropy.table.Table`
        the table of asset values as a function of time
    packetGainsT : `astropy.table.Table`
        gains table, if None, first transition year
    totalGainsT : `astropy.table.Table`
        total gains per tax year per asset
    firstMMTaxYear : `bool`
        Must be True for the first tax year when timely MM election is made.


    urevInc is per asset and changes with each packet.
    ``oBasis``, ``MMBasis`` are per packet (consider buying more over time,
    then some packet losses can be unrev but not all)
    """
    yearS = Time(f"{tyear:d}-1-1")
    yearE = Time(f"{tyear+1:d}-1-1")

    flt = packetsT['asset'] == assetId
    D = packetsT[flt]
    # In this tax year, only packetsT that are bought before the end of this year
    # and either not sold at all or sold during this year matter
    flt = (D['dateS'] < yearE) & (D['dateE'].mask | (D['dateE'] >= yearS))
    D = D[flt]
    D.add_column(D['dateE'].filled(fill_value=yearE), name='filled_dateE')
    D.sort(keys=['filled_dateE', 'dateS'])
    D.remove_column('filled_dateE')

    if firstMMTaxYear:
        totalGainsT = Table(
            names=('tyear', 'assetId', 'urevIncS', 'urevIncE',
                   'regInc', 'ltcGain'),
            dtype=(int, 'S20', float, float,
                   float, float))
        packetGainsT = Table(
            names=('tyear', 'assetId', 'packetId', 'oBasisS', 'MMBasisS',
                   'regInc', 'ltcGain', 'oBasisE', 'MMBasisE', 'urevInc'),
            dtype=(int, 'S20', 'S20', float, float,
                   float, float, float, float, float))
        # oBasisS : ordinary basis for this packet at the beginning of tax year
        # oBasisE : ordinary basis for this packet at the end of tax year
        urevIncS = 0
    else:
        prevY = tyear-1
        flt = (totalGainsT['assetId'] == assetId) & (totalGainsT['tyear'] == prevY)
        urevIncS = totalGainsT['urevIncE'][flt][0]

    urevInc = urevIncS
    yRegInc = 0.
    yLtcGain = 0.
    for pkt in D:
        # if pkt['dateS'] >= yearE or (not pkt['dateE'].mask and pkt['dateE'] < yearS):
        #     # We don't hold this packet during this tax year
        #     continue

        # This packet's contribution in this year
        dyRegInc = 0
        dyLtcGain = 0

        if pkt['dateS'] < yearS:
            # This packet was bought in an earlier year
            # overlapS = yearS
            if firstMMTaxYear:
                oBasisS = pkt['priceS']
                MMBasisS = oBasisS
                # Step up basis for first MM year
                yPriceS = getValue(yearS, priceT, 'date', 'price')
                if yPriceS > MMBasisS:
                    # transition rule, step up MM basis
                    MMBasisS = yPriceS
            else:
                # This packet's bases from previous tax year
                flt = ((packetGainsT['packetId'] == pkt['packetId'])
                       & (packetGainsT['tyear'] == prevY))
                oBasisS = packetGainsT['oBasisE'][flt][0]
                MMBasisS = packetGainsT['MMBasisE'][flt][0]
        else:
            # This packet was purchased during the MM tax year
            # overlapS = pkt['dateS']
            oBasisS = pkt['priceS']
            MMBasisS = oBasisS

        if pkt['dateE'].mask or pkt['dateE'] >= yearE:
            # This packet was not sold during this tax year
            overlapE = yearE
            priceE = getValue(overlapE, priceT, 'date', 'price')
        else:
            # This packet was sold, MM and ltcGain calculation
            overlapE = pkt['dateE']
            priceE = pkt['priceE']  # Here we know the exact price
        # Determine the MM regular income for the tax year
        dPrice = priceE - MMBasisS
        oBasisE = oBasisS
        MMBasisE = MMBasisS
        gain = dPrice * pkt['quantity']  # gain or loss
        dyRegInc = 0.
        if dPrice > 0:
            MMBasisE = MMBasisS + dPrice
            oBasisE = oBasisS + dPrice
            urevInc += gain
            dyRegInc = gain
        elif dPrice < 0 and urevInc > 0:
            lossLim = min(abs(gain), urevInc)  # positive
            dyRegInc = -lossLim
            urevInc -= lossLim
            dPrice2 = lossLim / pkt['quantity']
            MMBasisE = MMBasisS - dPrice2
            oBasisE = oBasisS - dPrice2
        # Determine the LtcGain of pre-MM years
        # if this packet is sold during the tax year
        dyLtcGain = 0.
        if not pkt['dateE'].mask and pkt['dateE'] < yearE:
            dPrice = priceE - oBasisE
            ltcGain = dPrice * pkt['quantity']  # gain or loss
            if dPrice > 0.:
                oBasisE += dPrice
                dyLtcGain = ltcGain
            elif dPrice < 0. and urevInc > 0.:
                print("WARNING ltcLoss should not happen")
                # lossLim = min(abs(ltcGain), urevInc)  # positive
                # dyLtcGain = -lossLim
                # urevInc -= lossLim
        packetGainsT.add_row(
            dict(tyear=tyear, assetId=assetId, packetId=pkt['packetId'],
                 oBasisS=oBasisS, MMBasisS=MMBasisS,
                 regInc=dyRegInc, ltcGain=dyLtcGain,
                 oBasisE=oBasisE, MMBasisE=MMBasisE,
                 urevInc=urevInc))

        yRegInc += dyRegInc
        yLtcGain += dyLtcGain

    totalGainsT.add_row(
        dict(tyear=tyear, assetId=assetId, urevIncS=urevIncS, urevIncE=urevInc,
             regInc=yRegInc, ltcGain=yLtcGain))

    return packetGainsT, totalGainsT
