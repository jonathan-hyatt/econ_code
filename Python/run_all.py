import multiprocessing as mp
import subprocess
import sys
from pprint import pprint
import traceback

num_min = 1440

def run_script(script_name, shared_num_min, new_db):
    subprocess.run(['python', script_name, str(shared_num_min.value), new_db]) 


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            pprint(self.name + " has failed.")
            tb = traceback.format_exc()
            self._cconn.close((e,tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


if __name__ == "__main__":


    new_db = ""
    if len(sys.argv) > 1:
        pprint("Creating new databases")
        new_db = "True"
    else:
        pprint("Using existing databases.")
        new_db = "False"


    #script_filenames = ["exchange_libs/bitget.py","exchange_libs/coinbase.py","exchange_libs/binance.py","exchange_libs/okx.py","exchange_libs/kraken.py","exchange_libs/bitget.py","exchange_libs/bitfinex.py","exchange_libs/gemini.py","exchange_libs/crypto.py","exchange_libs/htx.py","exchange_libs/mexc.py","exchange_libs/probit.py","exchange_libs/bitmart.py","exchange_libs/hotcoin.py","exchange_libs/toobit.py","exchange_libs/binanceUs.py","exchange_libs/coincheck.py","exchange_libs/phemex.py","exchange_libs/bitrue.py","exchange_libs/bitvenus.py","exchange_libs/bitflyer.py","exchange_libs/orangex.py","exchange_libs/digifinex.py","exchange_libs/hotscoin.py","exchange_libs/p2b.py","exchange_libs/bit_stamp.py"] 

    script_filenames = ["exchange_libs/bitget.py","exchange_libs/coinbase.py","exchange_libs/binance.py","exchange_libs/okx.py","exchange_libs/kraken.py","exchange_libs/bitfinex.py","exchange_libs/gemini.py","exchange_libs/crypto.py","exchange_libs/htx.py","exchange_libs/mexc.py","exchange_libs/probit.py","exchange_libs/bitmart.py","exchange_libs/hotcoin.py","exchange_libs/toobit.py","exchange_libs/binanceUs.py","exchange_libs/coincheck.py","exchange_libs/phemex.py","exchange_libs/bitrue.py","exchange_libs/bitvenus.py","exchange_libs/bitflyer.py","exchange_libs/orangex.py","exchange_libs/digifinex.py","exchange_libs/p2b.py","exchange_libs/bit_stamp.py"] 
    #script_filenames = ["exchange_libs/binance.py"]
    # "exchange_libs/gateio.py"   "exchange_libs/lbank.py"   --        Not working currently


    shared_num_min = mp.Value('d', num_min)







    # Create a process for each script
    processes = [Process(target=run_script, args=(script,shared_num_min, new_db), name=script) for script in script_filenames]

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All scripts have finished.")

#price and size are flipped on bitfinex
