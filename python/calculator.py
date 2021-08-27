from astropy.time import Time
from astropy.table import Table, MaskedColumn, Column
import numpy as np
import astropy.table


class MMCalculator:
    """The calculator"""

    def __init__(self, assetId, priceT=None, currName=None, xchgT=None):
        """References to given tables

        xchgT:
            ``date``, ``perCURR``
        """
        self.xchgT = xchgT
        self.currName = currName  # Prefix for currency
        self.assetId = assetId
        self.priceT = priceT
        self.assetId = None
        self.packetsT = None
        self.packetGainsT = None
        self.totalGainsT = None

    @staticmethod
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
                raise ValueError(f"Value data starts later than interested point {t}, keys: {k_t} {k_val}")
                return valueT[k_val][0]
            vlast = valueT[k_val][i-1]
            r = (t-valueT[k_t][i-1]) / (valueT[k_t][i]-valueT[k_t][i-1])
            return vlast + (valueT[k_val][i]-vlast) * r.value
        else:
            raise ValueError(f"Value data ends earlier than interested point {t}, keys: {k_t} {k_val}")
            return valueT[k_val][-1]

    def getPriceConverted(self, t):
        v = self.getValue(t, self.priceT, 'date', 'price')
        if self.currName is not None:
            xchg = self.getValue(t, self.xchgT, 'date', f'per{self.currName}')
        else:
            xchg = 1.
        return v*xchg

    @staticmethod
    def prepareRawTransactions(traw):
        """In-place modification of raw loaded csv table"""
        traw['date'] = Time(traw['date'])
        traw.rename_column('date', 'dateS')
        traw.add_column(MaskedColumn(traw['dateS'], name='dateE', mask=True))
        # traw.add_column(np.arange(1, len(traw)+1), name='pId')
        return traw

    def extendPriceTable(self, preprawT):
        """Adds new entries from the packetized transactions to the price table
        """
        p2 = preprawT.copy()
        p2.rename_column('dateS', 'date')
        p2.rename_column('unit price', 'price')
        R = astropy.table.vstack([self.priceT, p2])
        self.priceT = astropy.table.unique(R, keys=['date', 'price'], keep='first')

    def packetizeTransactions(self, traw, assetPrefix):
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
        traw = traw[traw['assetId'] == self.assetId]
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
        self.packetsT = packets

    def calculateOneYearOneAssetTaxes(self, tyear, firstMMTaxYear=False):
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
        packetsT = self.packetsT
        assetId = self.assetId

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
                names=('tyear', 'assetId',
                       'regInc', 'ltcGain', 'L10a_valueE', 'L10b_MMbasis', 'L10c_gainloss',
                       'L11_urevinc', 'L12_limitLoss'),
                dtype=(int, 'S20', float, float,
                       float, float, float, float, float, float, float))
            packetGainsT = Table(
                names=('tyear', 'assetId', 'packetId', 'oBasisS', 'MMBasisS', 'urevIncS'
                       'valueE', 'regInc', 'oBasisL', 'ltcGain', 'oBasisE', 'MMBasisE', 'urevIncE', 'soldThisYear'),
                dtype=(int, 'S20', 'S20', float, float, float,
                       float, float, float, float, float, float, float, bool))
            # oBasisS : ordinary basis cost for this packet at the beginning of tax year
            # oBasisE : ordinary basis cost for this packet at the end of tax year if not sold
            # valueE : FMV at the end of tax year or when sold
            # regInc: signed: ordinary income or allowed loss by urevinc
            # oBasisL: basis for long term capital gain, loss if sold
            # (oBasisS + tax year MM inclusion/allowed deduction)
            # ltcGain: signed: long term capital gain or loss if sold
            # urevIncE : unreversed inclusion if not sold
            urevIncS = 0
        else:
            prevY = tyear-1
            flt = (totalGainsT['assetId'] == assetId) & (totalGainsT['tyear'] == prevY)
            urevIncS = totalGainsT['urevIncE'][flt][0]

        yRegInc = 0.
        yLtcGain = 0.
        yearPacketsT = Table(
            names=('tyear', 'assetId', 'packetId', 'oBasisS', 'MMBasisS',
                   'valueE', 'regInc', 'ltcGain', 'oBasisE', 'MMBasisE', 'soldThisYear'),
            dtype=(int, 'S20', 'S20', float, float,
                   float, float, float, float, float, bool))
        for pkt in D:
            # This packet's contribution in this year
            dyRegInc = 0
            dyLtcGain = 0

            if pkt['dateS'] < yearS:
                # This packet was bought in an earlier year
                if firstMMTaxYear:
                    oBasisS = pkt['priceS']*pkt['quantity']
                    MMBasisS = oBasisS
                    # Step up basis for first MM year
                    yBasis = getValue(yearS, priceT, 'date', 'price')*pkt['quantity']
                    if yBasis > MMBasisS:
                        # transition rule, step up MM basis
                        MMBasisS = yBasis
                else:
                    # This packet's bases from previous tax year
                    flt = ((packetGainsT['packetId'] == pkt['packetId'])
                           & (packetGainsT['tyear'] == prevY))
                    oBasisS = packetGainsT['oBasisE'][flt][0]
                    MMBasisS = packetGainsT['MMBasisE'][flt][0]
            else:
                # This packet was purchased during the MM tax year
                oBasisS = pkt['priceS']*pkt['quantity']
                MMBasisS = oBasisS

            if pkt['dateE'].mask or pkt['dateE'] >= yearE:
                # This packet was not sold during this tax year
                soldThisYear = False
                overlapE = yearE
                valueE = getValue(overlapE, priceT, 'date', 'price')*pkt['quantity']
            else:
                # This packet was sold, here we get the MM
                # below the ltcGain
                soldThisYear = True
                overlapE = pkt['dateE']
                valueE = pkt['priceE']*pkt['quantity']  # Here we know the exact price
            # Determine the MM regular income/loss for the tax year
            gain = valueE - MMBasisS
            oBasisE = oBasisS
            MMBasisE = MMBasisS
            dyRegInc = 0.
            MMBasisE = MMBasisS + gain
            oBasisE = oBasisS + gain
            dyRegInc = gain
            # Determine the LtcGain of pre-MM years
            # if this packet is sold during the tax year
            # If packet was purchased this year, it'll be zero anyway
            dyLtcGain = 0.
            if soldThisYear:
                dyLtcGain = valueE - oBasisE
                oBasisE = valueE
            yearPacketsT.add_row(
                dict(tyear=tyear, assetId=assetId, packetId=pkt['packetId'],
                     oBasisS=oBasisS, MMBasisS=MMBasisS,
                     valueE=valueE, regInc=dyRegInc, ltcGain=dyLtcGain,
                     oBasisE=oBasisE, MMBasisE=MMBasisE,
                     soldThisYear=soldThisYear))

        # Now see how much loss we can afford with the urevinc
        yearPacketsT.sort(keys=['regInc', ], reverse=True)  # First the gains then the losses
        urevs = urevIncS + np.cumsum(yearPacketsT['regInc'])
        i_negs = np.flatnonzero(urevs < 0.)
        if len(i_negs) > 0:
            for ii in range(len(i_negs)):
                fi = i_negs[ii]
                if ii == 0:
                    # The first item that was not fully covered by urevinc
                    lossAdjust = abs(urevs[fi])
                else:
                    # All others must be totally reversed
                    lossAdjust = abs(yearPacketsT['regInc'][fi])
                yearPacketsT['regInc'][fi] += lossAdjust
                yearPacketsT['MMBasisE'][fi] += lossAdjust
                yearPacketsT['oBasisE'][fi] += lossAdjust
                if yearPacketsT['soldThisYear'][fi]:
                    dyLtcGain -= lossAdjust
                    yearPacketsT['oBasisE'][fi] -= lossAdjust
        # Calculcate the line values
        flt = yearPacketsT['soldThisYear']
        soldT = yearPacketsT[flt]
        keptT = yearPacketsT[np.logical_not(flt)]
        urevInc = urevIncS
        L10a_valueE = np.sum(keptT['valueE'])
        L10b_MMBasis = np.sum(keptT['MMBasisS'])
        L10c_gainloss = L10a_valueE - L10b_MMBasis
        L13c_gainloss = np.sum(soldT['valueE']) - np.sum(soldT['MMBasisS'])
        if L10c_gainloss < 0.:
            if L13c_gainloss > 0.:
                # It is possible that the sold gain is also used here in reversal
                urevInc += L13c_gainloss
            L11_urevInc = urevInc

            urevInc += L10c_gainloss

        if L13c_gainloss < 0. :
            if L10c_gainloss > 0.:
                # It is possible that the kept gain is also used here in reversal
                urevInc += L10c_gainloss
            L14a_urevInc = urevInc

        packetGainsT = astropy.table.vstack([packetGainsT, yearPacketsT], join_type='exact')
        # ======
        totalGainsT.add_row(
            dict(tyear=tyear, assetId=assetId, urevIncS=urevIncS, urevIncE=urevInc,
                 regInc=yRegInc, ltcGain=yLtcGain))

        return packetGainsT, totalGainsT
