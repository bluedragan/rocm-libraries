import pytest
from unittest.mock import MagicMock

from rocisa.instruction import SWaitCnt

from Tensile.Components.CustomSchedule import hasCustomSchedule, ScheduleInfo, verifyAscendingOrder, verifyLRsDoneInTime
from Tensile.Common import IsaVersion

# Helper to create a mock data type
def _mock_dtype(is_16bit=False, is_8bit=False, num_bytes=4):
    mock = MagicMock()
    mock.isHalf.return_value = is_16bit
    mock.isBFloat16.return_value = False # Assuming isHalf is enough for is16bit
    mock.isInt8.return_value = is_8bit
    mock.is8bitFloat.return_value = False # Assuming isInt8 is enough for is8bit
    mock.numBytes.return_value = num_bytes
    return mock

# Base kernel configuration factory
def create_base_kernel():
    kernel = {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "ISA": IsaVersion(9,5,0),
        "ProblemType": {
            "DataType": _mock_dtype(),
            "DataTypeA": _mock_dtype(),
            "DataTypeB": _mock_dtype(),
            "TransposeA": False,
            "TransposeB": False,
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 0,
        "PrefetchGlobalRead": 0, "PrefetchLocalRead": 0, "DirectToLds": False,
        "GlobalReadVectorWidthA": 0, "GlobalReadVectorWidthB": 0,
        "LocalReadVectorWidth": 0,
        "MatrixInstruction": [],
        "MIWaveGroup": [],
        "LDSTrInst": False,
        "TransposeLDS": 0,
        "ForceUnrollSubIter": False,
        "SwapGlobalReadOrder": False, # For asserting it gets set
        "UsePLRPack": False, # For asserting it gets set
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
    }
    return kernel

class TestCustomSchedule:
    def test_no_custom_schedule(self):
        """Test that a kernel that doesn't match any condition returns False."""
        kernel = create_base_kernel()
        # An empty kernel should not have a custom schedule
        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert not has_schedule
        assert schedule_info is None


    def test_schedule_256x256x64_16bit_TN(self):
        """Tests the 256x256x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2], "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 128
        valid, message = schedule_info.isValid({"kernel" : kernel})
        assert valid, message
        assert 'PackA0' not in schedule_info.optSchedule
        assert not kernel["UsePLRPack"]

    def test_schedule_256x256x64_16bit_NT(self):
        """Tests the 256x256x64 16-bit NT schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": False, "TransposeB": True
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": False, "TransposeLDS": 0, "MIWaveTileA": 8, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 128
        assert 'PackA0' in schedule_info.optSchedule
        assert kernel["UsePLRPack"]
        valid, message = schedule_info.isValid({"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(False, False), (True, True)])
    def test_schedule_256x256x64_16bit_NN_TT(self, transA, transB):
        """Tests the 256x256x64 16-bit NN and TT schedules."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 128
        assert kernel["UsePLRPack"]
        valid, message = schedule_info.isValid({"kernel" : kernel})
        assert valid, message
        if transA and transB: # isTT
            assert kernel["SwapGlobalReadOrder"]
            assert 'PackB0' in schedule_info.optSchedule
            assert 'PackA0' not in schedule_info.optSchedule
        else: # isNN
            assert not kernel["SwapGlobalReadOrder"]
            assert 'PackA0' in schedule_info.optSchedule
            assert 'PackB0' not in schedule_info.optSchedule

    def test_schedule_256x256x128_8bit_TN(self):
        """Tests the 256x256x128 8-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_8bit = _mock_dtype(is_8bit=True, num_bytes=1)
        kernel["ProblemType"].update({
            "DataType": dtype_8bit, "DataTypeA": dtype_8bit, "DataTypeB": dtype_8bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 128,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0, "DirectToLds": True,
            "GlobalReadVectorWidthA": 16, "GlobalReadVectorWidthB": 16, "LocalReadVectorWidth": 16,
            "MatrixInstruction": [16,16,128,1], "MIWaveGroup": [2,2], "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 64
        assert len(schedule_info.mfmaReorder) > 0
        valid, message = schedule_info.isValid({"kernel" : kernel})
        assert valid, message

    def test_schedule_192x256x64_16bit_NN(self):
        """Tests the 192x256x64 16-bit NN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": False, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 192, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": True, "TransposeLDS": 1, "MIWaveTileA": 6, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 96
        assert kernel["SwapGlobalReadOrder"]
        valid, message = schedule_info.isValid({"kernel" : kernel})
        assert valid, message


class TestCustomScheduleValidation:
    def test_schedule_validation_non_descending_order(self):
        """
        Test of the rule that instructions in each category
        appear in non-descending order
        """

        sched = ScheduleInfo(
            None, None, {"P": [[3, 2, 1]]}, None, None, None, None
        )
        status, message = verifyAscendingOrder(sched)

        expected = "Non-descending-order rule failed, schedule key 'P', sequence [3, 2, 1]: value 2 at index 1 is less than 3 at index 0."
        assert status == False
        assert message == expected

        sched = ScheduleInfo(
            None, None, {"P": [[1, 1, 2]]}, None, None, None, None
        )
        status, message = verifyAscendingOrder(sched)
        assert status == True

    def test_schedule_validation_disable(self):
        """
        Test of the flag that custom mainloop schedule (CMS) developers can use to override the
        validation checks.
        """
        kernel = create_base_kernel()
        invalid_schedule = {"P": [[3, 2, 1]]}

        # No verification message means that the schedule info is considered valid.
        scheduleInfo = ScheduleInfo(
            None, None, invalid_schedule, None, None, None, None
        )
        scheduleInfo.disableValidation()
        status, message = scheduleInfo.isValid({"kernel" : {"DepthU": 42}})
        assert status == True
        assert message == "CMS validation explicitly disabled. Running on kernel with MT0xMT1xDepthU = ?x?x42"

        # A non-empty verification message means that the schedule info is considered invalid.
        status, message = ScheduleInfo(
            None, None, invalid_schedule, None, None, None, None
        ).isValid({})
        assert status == False


class TestVerifyLRsDoneInTime:
    def test_simple_LR0(self):
        """
        Verify the simple case where both LRA0 and LRB0 are issued and finished before the halfway point of the main loop.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]
        
        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]]
        }

        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        optSchedule["LRA0"] = [[1, 6]]
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert not status, f"Schedule should have failed (LRA0 issued after halfway point), but passed."

        optSchedule["LRA0"] = [[1, 2]]
        optSchedule["LRB0"] = [[3, 6]]
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert not status, f"Schedule should have failed (LRB0 issued after halfway point), but passed."

    def test_simple_LR0_w_LR1(self):
        """
        Handle case where we start reading LRA1 before halfway point.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[3, 7]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[2]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation 1/2 but did not. {message}"

        # Changing barrier from 0 to 1 for LRA1 should still pass
        syncCode[0].dscnt = 1
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation 2/2 but did not. {message}"

    
    def test_complex_LR0(self):
        """
        2nd LRB0 is not needed until iteration 6 & 7, can have SWaitCnt for it after the halfway point.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[3, 4]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 2]],  # 2nd LRB0 is n
        }
        syncCode = [
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"


    def test_simple_LR1(self):
        """
        Case where LR1 is finished before the end of the current iteration.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[1, 7]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[4, 4]],
            "LRB1": [[4, 4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

    def test_pre_loop_SWaitCnt(self):
        """
        Case where LR1 is finished before start of next iteration.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[-1, 3]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[5, 5]],
            "LRB1": [[6, 6]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"


    def test_pre_loop_LR(self):
        """
        Case where an LR is issued before the start of the loop (idx=-1).
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[3]],
            "LRA0": [[-1, -1]],
            "LRB0": [[-1, -1]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="")
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

    def test_simple_LR1_never_guaranteed(self):
        """
        Case where LR1 is finished before the end of loop.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[1]],
            "LRA0": [[0, 0]],
            "LRB0": [[0, 0]],
            "LRA1": [[4, 4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment=""),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert not status, f"Schedule should have failed (LRA1 never guaranteed), but passed. {message}"

    def test_complex_LR1(self):
        """
        Case where LR1 finishes during the beginning of next iteration.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[1, 3, 7]],
            "LRA0": [[2, 2]],
            "LRB0": [[2, 2]],
            "LRA1": [[4, 4]],
            "LRB1": [[4, 4]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/2 LRB1"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="All of LRA0 and LRB0"),
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="2/2 LRA1 and 1/2 LRB1"),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

    def test_more_LRs(self):
        """
        Case where each LR reads less than 1 WaveTile worth of data.
        """
        kernel = create_base_kernel()
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[0, 1, 3, 4, 5, 7]],
            "LRA0": [[0, 0, 3, 3]],
            "LRB0": [[1, 1, 2, 4]],
            "LRA1": [[5, 5, 7, 7]],
            "LRB1": [[6, 6, 7, 7]],
        }
        syncCode = [
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="4/4 LRA1 and 2/4 LRB1"),
            SWaitCnt(dscnt=2, vlcnt=-1, vscnt=-1, comment="4/4 LRB1"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/4 LRA0 and 3/4 LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="4/4 LRA0 and 3/4 LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="4/4 LRB0"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/4 LRA1 and 2/4 LRB1"),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

    
    def test_less_LRs(self):
        """
        Case where each LR reads more than 1 WaveTile worth of data.
        """
        kernel = create_base_kernel()
        kernel["MIWaveTileA"] = 4
        kernel["MIWaveTileB"] = 4
        num_vmfma = 2 * kernel["MIWaveTileA"] * kernel["MIWaveTileB"]

        optSchedule = {
            "SYNC": [[7, 15, 31]],
            "LRA0": [[12, 13]],
            "LRB0": [[13, 14]],
            "LRA1": [[16, 17]],
            "LRB1": [[18, 19]],
        }
        syncCode = [
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="2/2 LRB1"),
            SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="LRA0 and LRB0"),
            SWaitCnt(dscnt=1, vlcnt=-1, vscnt=-1, comment="LRA1 and 1/2 LRB1"),
        ]
        sched = ScheduleInfo(1, num_vmfma, optSchedule, syncCode, None, None, None)
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert status, f"Schedule should have passed validation but did not. {message}"

        # Failure case
        optSchedule["SYNC"][0][0] = 8
        status, message = verifyLRsDoneInTime(sched, {"kernel": kernel})
        assert not status, f"Schedule should have failed (LRB0 not finished before being needed), but passed. {message}"
