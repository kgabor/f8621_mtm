from astropy.time import Time
from astropy.table import Table, MaskedColumn, Column
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
        return vlast + (valueT[k_val][i]-vlast) * (t-valueT[k_t][i-1]) / (valueT[k_t][i]-valueT[k_t][i-1])
    else:
        print("WARNING: Value data ends earlier than interested dates")
        return valueT[k_val][-1]


def prepareRawTransactions(traw):
    """In-place modification of raw loaded csv table"""
    traw['date'] = Time(traw['date'])
    traw.rename_column('date', 'dateS')
    traw.add_column(MaskedColumn(traw['dateS'], name='dateE', mask=True))
    # traw.add_column(np.arange(1, len(traw)+1), name='pId')
    return traw


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
    """
    packets = traw[traw['type'] == 'buy']
    packets.sort(keys=['dateS', ])
    packets.remove_columns(['type', 'total price'])
    packets.rename_columns(['unit price'], ['priceS'])
    packets.add_column(MaskedColumn(np.zeros(len(packets)), name='priceE', mask=True))
    packets.add_column(Column([f'{assetPrefix}{x:03d}' for x in range(1, len(packets)+1)],
                              dtype='S20'), name='packetId')

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
            packets['priceE'].mask[irec] = False
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
                packets['packetId'][irec] = npId
                npId += 1
                saleUnits = 0.
            ii += 1
    return packets


def calculateOneYearOneAssetTaxes(packets, assetId, tyear, priceT,
                                  packetGainsT=None, totalGainsT=None,
                                  firstMMTaxYear=False):
    """
    Must be used in incrementing tax years.
    Goes through the packets and calculates either its tax year MM gain,

    Parameters
    ==========
    packets: `astropy.table.Table`
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

    flt = packets['asset'] == assetId
    D = packets[flt]
    # In this tax year, only packets that are bought before the end of this year
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
        flt = (totalGainsT['asset'] == assetId) & (totalGainsT['year'] == prevY)
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
        else:
            # This packet was sold, MM and ltcGain calculation
            overlapE = pkt['dateE']
        # Determine the MM regular income for the tax year
        priceE = getValue(overlapE, priceT, 'date', 'price')
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

        # print(f"{tyear:4d} {pkt['pId']} : {dTotGain:6.0f} {dyGain:6.0f} {dUrevInc:6.0f} | "
        #       f"{totGain:6.0f} {yRegInc:6.0f} {urevInc:6.0f}")
    totalGainsT.add_row(
        dict(tyear=tyear, assetId=assetId, urevIncS=urevIncS, urevIncE=urevInc,
             regInc=yRegInc, ltcGain=yLtcGain))

    return packetGainsT, totalGainsT
