from astropy.time import Time
from astropy.table import Table, MaskedColumn, Column
import numpy as np
import astropy.table
import astropy.units as units

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
        self.priceT = priceT[priceT['assetId'] == assetId]
        self.packetsT = None
        self.packetGainsT = None
        # self.totalGainsT = None

    # @staticmethod
    # def getValue(t, valueT, k_t, k_val):
    #     """

    #     Interpolates
    #     Parameters
    #     ----------
    #     t  : value(s) of key where we are interested in
    #     k_t : key for t in valueT for ``t``
    #     k_val : key for value in valueT
    #     """
    #     if not np.all(valueT[k_t][1:] - valueT[k_t][:-1] >= 0):
    #         raise ValueError("valueT is not monotonic increasing in dates")
    #     if np.isscalar(t):
    #         singleVal = True
    #         t = np.array([t, ], dtype=float)
    #     else:
    #         singleVal = False
    #     i = np.searchsorted(valueT[k_t], t)
    #     if np.any(i) >= len(valueT):
    #         badt = t[i >= len(valueT)]
    #         raise ValueError(f"Value data ends earlier than interested point {badt}, keys: {k_t} {k_val}")
    #     if i == 0:
    #         raise ValueError(f"Value data starts later than interested point {t}, keys: {k_t} {k_val}")

    #     r = np.zeros_like(t, dtype=float)
    #     eqflt = valueT[k_t][i] == t
    #     r[eqflt] = valueT[k_val][i][eqflt]
    #     interpflt = np.logical_not(eqflt)

    #     vlast = valueT[k_val][i-1][interpflt]
    #     ratio = (t-valueT[k_t][i-1][interpflt]) / (valueT[k_t][i][interpflt]
    #                                                - valueT[k_t][i-1][interpflt])
    #     r[interpflt] = vlast + (valueT[k_val][i][interpflt]-vlast) * ratio.value

    #     if singleVal:
    #         return r[0]
    #     else:
    #         return r

    @staticmethod
    def getValue(t, valueT, k_t, k_val, tmax=None):
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
            d = (valueT[k_t][i]-valueT[k_t][i-1])
            r = (t-valueT[k_t][i-1]) / d
            if tmax is not None and abs(r.value-round(r.value)) * d > tmax:
                print(f"WARNING: interpolation exceeds {tmax} distance")
            return vlast + (valueT[k_val][i]-vlast) * r.value
        else:
            print("WARNING: Value data ends earlier than interested dates")
            return valueT[k_val][-1]

    @staticmethod
    def getMultiValue(tstamps, valueT, k_t, k_val, tmax=None):
        """Get multiple values from a price or exchange rate table
        """
        v = np.zeros(len(tstamps), dtype=float)
        for i, t in enumerate(tstamps):
            v[i] = MMCalculator.getValue(t, valueT, k_t, k_val, tmax=tmax)
        return v

    # def getMultiPrice(self, tstamps):
    #     """Get the price in USD
    #     """
    #     v = self.getMultiValue(tstamps, self.priceT, 'date', 'price')
    #     if self.currName is not None:
    #         xchg = self.getMultiValue(tstamps, self.xchgT, 'date', f'per{self.currName}')
    #     else:
    #         xchg = 1.
    #     return v*xchg

    def getMultiRates(self, tstamps, tmax=62*units.day):
        rates = self.getMultiValue(tstamps, self.xchgT, 'date', f'per{self.currName}', tmax=tmax)
        return rates

    def getPrice(self, t):
        """Get the price in USD
        """
        v = self.getValue(t, self.priceT, 'date', 'price', tmax=30*units.day)
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
        return packets

    def convertPricesToUSD(self):
        """Convert all packetized transactions to USD
        """
        if self.xchgT is None:
            print("WARNING No exchange rates table, leaving unchanged.")
            return
        rates = self.getMultiRates(self.packetsT['dateS'])
        self.packetsT['priceS'] = self.packetsT['priceS'] * rates

        validPriceE = np.logical_not(self.packetsT['priceE'].mask)
        rates = self.getMultiRates(self.packetsT['dateE'][validPriceE])
        self.packetsT['priceE'][validPriceE] = self.packetsT['priceE'][validPriceE] * rates

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

        flt = packetsT['assetId'] == assetId
        D = packetsT[flt]
        # In this tax year, only packetsT that are bought before the end of this year
        # and either not sold at all or sold during this year matter
        flt = (D['dateS'] < yearE) & (D['dateE'].mask
                                      | (D['dateE'].filled(Time('1990-01-01')) >= yearS))
        D = D[flt]
        D.add_column(D['dateE'].filled(fill_value=yearE), name='filled_dateE')
        D.sort(keys=['filled_dateE', 'dateS'])
        D.remove_column('filled_dateE')

        if firstMMTaxYear:
            # totalGainsT = Table(
            #     names=('tyear', 'assetId', 'holdValueE', 'holdRegInc', 'soldValueE'
            #            'soldRegInc', 'ltcGain'),
            #     dtype=(int, 'S20', float, float, float,
            #            float, float))
            # holdRegInc, soldRegInc, ltcGain: signed, with allowed loss
            packetGainsT = Table(
                names=('tyear', 'assetId', 'packetId', 'oBasisS', 'MMBasisS', 'urevIncS',
                       'valueE', 'holdRegInc', 'holdRegGain', 'holdRegLoss',
                       'soldRegGain', 'soldRegLoss', 'oBasisL', 'ltcGain',
                       'oBasisE', 'MMBasisE', 'urevIncE', 'soldOtherLoss', 'soldThisYear'),
                dtype=(int, 'S20', 'S20', float, float, float,
                       float, float, float, float,
                       float, float, float, float,
                       float, float, float, float, bool))
            self.packetGainsT = packetGainsT
            # oBasisS : ordinary basis cost for this packet at the beginning of tax year
            # oBasisE : ordinary basis cost for this packet at the end of tax year if not sold
            # valueE : FMV at the end of tax year or when sold
            # regInc: signed: ordinary income or allowed loss by urevinc
            # oBasisL: basis for long term capital gain, loss if sold
            # (oBasisS + tax year MM inclusion/allowed deduction)
            # ltcGain: signed: long term capital gain or loss if sold
            # urevIncE : unreversed inclusion if not sold
            prevY = None
        else:
            prevY = tyear-1
            # flt = (totalGainsT['assetId'] == assetId) & (totalGainsT['tyear'] == prevY)
            # urevIncS = totalGainsT['urevIncE'][flt][0]
            packetGainsT = self.packetGainsT

        # yRegInc = 0.
        # yLtcGain = 0.
        # yearPacketsT = Table(
        #     names=('tyear', 'assetId', 'packetId', 'oBasisS', 'MMBasisS',
        #            'valueE', 'regInc', 'ltcGain', 'oBasisE', 'MMBasisE', 'soldThisYear'),
        #     dtype=(int, 'S20', 'S20', float, float,
        #            float, float, float, float, float, bool))
        for pkt in D:
            # This packet's contribution in this year
            dholdRegInc = 0  # Signed value
            dholdRegGain = 0  # L10c
            dholdRegLoss = 0  # L12
            dsoldRegGain = 0  # L13c
            dsoldRegLoss = 0  # L14b
            dsoldOtherLoss = 0  # L14c
            dLtcGain = 0

            if pkt['dateS'] < yearS:
                # This packet was bought in an earlier year
                if firstMMTaxYear:
                    urevInc = 0.
                    oBasisS = pkt['priceS']*pkt['quantity']
                    if pkt['reinvest']:
                        # If packet was a result of reinvest
                        # we will pay ltc on this amount
                        oBasisS = 0.
                    MMBasisS = oBasisS
                    # Step up basis for first MM year
                    yBasis = self.getPrice(yearS)*pkt['quantity']
                    print(yBasis, MMBasisS)
                    if yBasis > MMBasisS:
                        # transition rule, step up MM basis
                        MMBasisS = yBasis
                else:
                    # This packet's bases from previous tax year
                    flt = ((packetGainsT['packetId'] == pkt['packetId'])
                           & (packetGainsT['tyear'] == prevY))
                    oBasisS = packetGainsT['oBasisE'][flt][0]
                    MMBasisS = packetGainsT['MMBasisE'][flt][0]
                    urevInc = packetGainsT['urevIncE'][flt][0]
            else:
                # This packet was purchased during the MM tax year
                oBasisS = pkt['priceS']*pkt['quantity']
                urevInc = 0.
                if pkt['reinvest']:
                    # If packet was a result of reinvest
                    # we pay ord. income tax on this amount
                    oBasisS = 0.
                MMBasisS = oBasisS
            urevIncS = urevInc
            if pkt['dateE'].mask or pkt['dateE'] >= yearE:
                # This packet was not sold during this tax year
                soldThisYear = False
                overlapE = yearE
                valueE = self.getPrice(overlapE)*pkt['quantity']
            else:
                # This packet was sold during this tax year
                soldThisYear = True
                if pkt['dateE'] - pkt['dateS'] >= 1. * units.year:
                    longTerm = True
                else:
                    longTerm = False

                overlapE = pkt['dateE']
                valueE = pkt['priceE']*pkt['quantity']  # Here we know the exact price
            # Determine the MM regular income for the tax year
            gain = valueE - MMBasisS
            oBasisE = oBasisS
            MMBasisE = MMBasisS
            if gain > 0:
                MMBasisE = MMBasisS + gain
                oBasisE = oBasisS + gain
                urevInc += gain
                if soldThisYear:
                    dsoldRegGain = gain
                    oBasisL = oBasisE
                    dLtcGain = valueE - oBasisL
                    # This packet has no further bases
                    oBasisE = 0
                    MMBasisE = 0
                else:
                    dholdRegInc = gain
                    dholdRegGain = gain
                    # No ltc calculation this year
                    oBasisL = 0

            else:  # Loss on MM tax year
                lossLim = min(abs(gain), urevInc)  # positive
                MMBasisE = MMBasisS - lossLim
                oBasisE = oBasisS - lossLim
                if soldThisYear:
                    dsoldRegLoss = lossLim
                    oBasisL = oBasisE
                    if abs(gain) > urevInc:
                        dsoldOtherLoss = abs(gain) - lossLim  # This is a long term cap. loss
                        oBasisL -= dsoldOtherLoss
                    dLtcGain = valueE - oBasisL  # We can still have a positive ltc
                    # dsoldOtherLoss and dLtcGain can net each other in schedule D
                    # but overall loss that can be accumulated is limited
                    oBasisE = 0
                    MMBasisE = 0
                    urevInc = 0
                else:
                    dholdRegInc = -lossLim
                    dholdRegLoss = lossLim
                    oBasisL = 0
                    urevInc -= lossLim

            if soldThisYear:
                urevIncE = 0.
            else:
                urevIncE = urevInc
            packetGainsT.add_row(
                dict(tyear=tyear, assetId=assetId, packetId=pkt['packetId'],
                     oBasisS=oBasisS, MMBasisS=MMBasisS, urevIncS=urevIncS,
                     valueE=valueE, holdRegInc=dholdRegInc, holdRegGain=dholdRegGain,
                     holdRegLoss=dholdRegLoss, soldRegGain=dsoldRegGain,
                     soldRegLoss=dsoldRegLoss,
                     oBasisL=oBasisL, ltcGain=dLtcGain, oBasisE=oBasisE, MMBasisE=MMBasisE,
                     urevIncE=urevIncE, soldOtherLoss=dsoldOtherLoss, soldThisYear=soldThisYear))

        #     yearPacketsT.add_row(
        #         dict(tyear=tyear, assetId=assetId, packetId=pkt['packetId'],
        #              oBasisS=oBasisS, MMBasisS=MMBasisS,
        #              valueE=valueE, regInc=dyRegInc, ltcGain=dyLtcGain,
        #              oBasisE=oBasisE, MMBasisE=MMBasisE,
        #              soldThisYear=soldThisYear))

        # # Now see how much loss we can afford with the urevinc
        # yearPacketsT.sort(keys=['regInc', ], reverse=True)  # First the gains then the losses
        # urevs = urevIncS + np.cumsum(yearPacketsT['regInc'])
        # i_negs = np.flatnonzero(urevs < 0.)
        # if len(i_negs) > 0:
        #     for ii in range(len(i_negs)):
        #         fi = i_negs[ii]
        #         if ii == 0:
        #             # The first item that was not fully covered by urevinc
        #             lossAdjust = abs(urevs[fi])
        #         else:
        #             # All others must be totally reversed
        #             lossAdjust = abs(yearPacketsT['regInc'][fi])
        #         yearPacketsT['regInc'][fi] += lossAdjust
        #         yearPacketsT['MMBasisE'][fi] += lossAdjust
        #         yearPacketsT['oBasisE'][fi] += lossAdjust
        #         if yearPacketsT['soldThisYear'][fi]:
        #             dyLtcGain -= lossAdjust
        #             yearPacketsT['oBasisE'][fi] -= lossAdjust
        # # Calculcate the line values
        # flt = yearPacketsT['soldThisYear']
        # soldT = yearPacketsT[flt]
        # keptT = yearPacketsT[np.logical_not(flt)]
        # urevInc = urevIncS
        # L10a_valueE = np.sum(keptT['valueE'])
        # L10b_MMBasis = np.sum(keptT['MMBasisS'])
        # L10c_gainloss = L10a_valueE - L10b_MMBasis
        # L13c_gainloss = np.sum(soldT['valueE']) - np.sum(soldT['MMBasisS'])
        # if L10c_gainloss < 0.:
        #     if L13c_gainloss > 0.:
        #         # It is possible that the sold gain is also used here in reversal
        #         urevInc += L13c_gainloss
        #     L11_urevInc = urevInc

        #     urevInc += L10c_gainloss

        # # ======
        # totalGainsT.add_row(
        #     dict(tyear=tyear, assetId=assetId, urevIncS=urevIncS, urevIncE=urevInc,
        #          regInc=yRegInc, ltcGain=yLtcGain))

        return packetGainsT

    def printFormStatement(self, tyear):
        J = astropy.table.join(self.packetGainsT[self.packetGainsT['tyear'] == tyear], self.packetsT,
                               keys=['packetId', 'assetId'])
        # Sold items
        J2 = J[J['soldThisYear']]
        J2['dateS'] = Time(J2['dateS'], out_subfmt='date')
        J2['dateE'] = Time(J2['dateE'], out_subfmt='date')
        H = J2['dateS', 'dateE', 'quantity', 'valueE', 'MMBasisS', 'soldRegGain', 'urevIncS',
               'soldRegLoss', 'soldOtherLoss', 'oBasisL', 'ltcGain']
        H.sort(keys=['dateE'])
        print("Sales during the tax year\n")
        astropy.io.ascii.write(
            H, format='fixed_width_two_line',
            formats={'MMBasisS': "%.2f", 'valueE': "%.2f",
                     'oBasisL': "%.2f", 'ltcGain': "%.2f", 'urevIncS': "%.2f", 'soldRegLoss': "%.2f",
                     'soldRegGain': "%.2f", 'soldOtherLoss': "%.2f"})

        snames = ('quantity', 'valueE', 'MMBasisS', 'soldRegGain', 'soldRegLoss', 'soldOtherLoss', 'ltcGain')
        print("\nTotals:")
        print("=======")
        for x in snames:
            s = np.sum(J2[x])
            print(f"{x:15s}: {s:8.2f}")

        print("\n\n")
        # ==========

        J2 = J[np.logical_not(J['soldThisYear'])]
        J2['dateS'] = Time(J2['dateS'], out_subfmt='date')
        H = J2['dateS', 'quantity', 'valueE', 'MMBasisS', 'holdRegGain', 'urevIncS', 'holdRegLoss']
        print("Holdings at the end of the tax year\n")
        astropy.io.ascii.write(
            H, format='fixed_width_two_line',
            formats={'quantity': '%.2f', 'MMBasisS': "%.2f", 'valueE': "%.2f",
                     'urevIncS': "%.2f", 'holdRegLoss': "%.2f",
                     'holdRegGain': "%.2f"})

        snames = ('quantity', 'valueE', 'MMBasisS', 'urevIncS', 'holdRegGain', 'holdRegLoss','holdRegInc')
        print("\nTotals:")
        print("=======")
        for x in snames:
            s = np.sum(J2[x])
            print(f"{x:15s}: {s:8.2f}")
