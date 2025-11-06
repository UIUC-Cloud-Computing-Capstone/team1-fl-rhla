"""
Unit tests for log_training_time function in ffm_fedavg_depthffm_fim.py
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import time
import sys
import os
import re

# Add parent directory to path to import the function
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.engine.ffm_fedavg_depthffm_fim import log_training_time


class TestLogTrainingTime(unittest.TestCase):
    """Test cases for log_training_time function"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock args object with logger
        self.mock_args = Mock()
        self.mock_logger = Mock()
        self.mock_args.logger = self.mock_logger
        
        # Set initial training start time
        self.training_start_time = time.time()

    def test_target_acc_none_defaults_to_0_7(self):
        """Test that target_acc defaults to 0.7 when None"""
        self.mock_args.target_acc = None
        
        best_test_acc = 0.5
        first_time_reaching_target_acc = False
        
        result = log_training_time(
            self.mock_args,
            self.training_start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify target_acc was set to 0.7
        self.assertEqual(self.mock_args.target_acc, 0.7)
        
        # Verify logger was called with default value message
        self.mock_logger.info.assert_any_call(
            'Target acc is not set, using default value: 0.7',
            main_process_only=True
        )
        
        # Verify general logging was called
        self.assertTrue(self.mock_logger.info.called)
        self.assertFalse(result)  # Should return False since acc < 0.7

    def test_target_acc_set_explicitly(self):
        """Test that explicitly set target_acc is used"""
        self.mock_args.target_acc = 0.8
        
        best_test_acc = 0.75
        first_time_reaching_target_acc = False
        
        result = log_training_time(
            self.mock_args,
            self.training_start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify target_acc was not changed
        self.assertEqual(self.mock_args.target_acc, 0.8)
        
        # Verify default value message was NOT called
        default_value_calls = [
            call for call in self.mock_logger.info.call_args_list
            if 'Target acc is not set' in str(call)
        ]
        self.assertEqual(len(default_value_calls), 0)
        
        # Verify general logging was called
        self.assertTrue(self.mock_logger.info.called)
        self.assertFalse(result)  # Should return False since 0.75 < 0.8

    def test_first_time_reaching_target_acc(self):
        """Test when best_test_acc reaches target_acc for the first time"""
        self.mock_args.target_acc = 0.75
        
        best_test_acc = 0.8
        first_time_reaching_target_acc = False
        
        result = log_training_time(
            self.mock_args,
            self.training_start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify result is True (first time reaching target)
        self.assertTrue(result)
        
        # Verify general logging was called
        self.assertTrue(self.mock_logger.info.called)
        
        # Verify "first time" message was logged
        first_time_calls = [
            call for call in self.mock_logger.info.call_args_list
            if 'first time that best_test_acc' in str(call)
        ]
        self.assertEqual(len(first_time_calls), 1)

    def test_already_reached_target_acc(self):
        """Test when target_acc was already reached before"""
        self.mock_args.target_acc = 0.75
        
        best_test_acc = 0.8
        first_time_reaching_target_acc = True  # Already reached
        
        result = log_training_time(
            self.mock_args,
            self.training_start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify result is still True
        self.assertTrue(result)
        
        # Verify "first time" message was NOT logged again
        first_time_calls = [
            call for call in self.mock_logger.info.call_args_list
            if 'first time that best_test_acc' in str(call)
        ]
        self.assertEqual(len(first_time_calls), 0)

    def test_best_test_acc_below_target(self):
        """Test when best_test_acc is below target_acc"""
        self.mock_args.target_acc = 0.8
        
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        result = log_training_time(
            self.mock_args,
            self.training_start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify result is False (not reached)
        self.assertFalse(result)
        
        # Verify "first time" message was NOT logged
        first_time_calls = [
            call for call in self.mock_logger.info.call_args_list
            if 'first time that best_test_acc' in str(call)
        ]
        self.assertEqual(len(first_time_calls), 0)

    def test_best_test_acc_exactly_at_target(self):
        """Test when best_test_acc exactly equals target_acc"""
        self.mock_args.target_acc = 0.75
        
        best_test_acc = 0.75
        first_time_reaching_target_acc = False
        
        result = log_training_time(
            self.mock_args,
            self.training_start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify result is True (reached target)
        self.assertTrue(result)
        
        # Verify "first time" message was logged
        first_time_calls = [
            call for call in self.mock_logger.info.call_args_list
            if 'first time that best_test_acc' in str(call)
        ]
        self.assertEqual(len(first_time_calls), 1)

    def test_training_time_calculation(self):
        """Test that training time is calculated correctly"""
        self.mock_args.target_acc = 0.5
        
        # Set training start time to 10 seconds ago
        training_start = time.time() - 10
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        result = log_training_time(
            self.mock_args,
            training_start,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify general logging was called with time information
        general_calls = [
            call for call in self.mock_logger.info.call_args_list
            if 'training time:' in str(call)
        ]
        self.assertGreater(len(general_calls), 0)
        
        # Verify the logged message contains time information
        logged_message = str(self.mock_logger.info.call_args_list)
        self.assertIn('training time:', logged_message)
        self.assertIn('seconds', logged_message)
        self.assertIn('minutes', logged_message)

    def test_logger_called_with_main_process_only(self):
        """Test that logger.info is called with main_process_only=True"""
        self.mock_args.target_acc = 0.5
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            self.training_start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify all logger.info calls have main_process_only=True
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            self.assertIn('main_process_only', kwargs)
            self.assertTrue(kwargs['main_process_only'])

    def test_multiple_calls_tracking_first_time(self):
        """Test that first_time_reaching_target_acc is properly tracked across calls"""
        self.mock_args.target_acc = 0.8
        
        # First call: below target
        result1 = log_training_time(
            self.mock_args,
            self.training_start_time,
            0.6,  # below target
            False
        )
        self.assertFalse(result1)
        
        # Second call: reaches target for first time
        result2 = log_training_time(
            self.mock_args,
            self.training_start_time,
            0.85,  # above target
            result1  # pass previous result
        )
        self.assertTrue(result2)
        
        # Third call: still above target, but already reached
        result3 = log_training_time(
            self.mock_args,
            self.training_start_time,
            0.9,  # still above target
            result2  # pass previous result (True)
        )
        self.assertTrue(result3)
        
        # Verify "first time" message was only logged once
        first_time_calls = [
            call for call in self.mock_logger.info.call_args_list
            if 'first time that best_test_acc' in str(call)
        ]
        self.assertEqual(len(first_time_calls), 1)

    @patch('algorithms.engine.ffm_fedavg_depthffm_fim.time.time')
    def test_training_time_calculation_exact_seconds(self, mock_time):
        """Test that training time is calculated exactly as end_time - start_time"""
        self.mock_args.target_acc = 0.5
        
        # Set up mock time to return specific values
        start_time = 1000.0
        end_time = 1010.5
        expected_time = end_time - start_time  # 10.5 seconds
        
        mock_time.return_value = end_time
        
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify the logged message contains the exact time
        general_log_call = None
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'training time:' in args[0] and 'first time' not in args[0]:
                general_log_call = args[0]
                break
        
        self.assertIsNotNone(general_log_call)
        # Extract the time value from the log message
        time_match = re.search(r'training time: ([\d.]+) seconds', general_log_call)
        self.assertIsNotNone(time_match)
        logged_seconds = float(time_match.group(1))
        
        # Verify the logged time matches expected calculation
        self.assertAlmostEqual(logged_seconds, expected_time, places=2)

    @patch('algorithms.engine.ffm_fedavg_depthffm_fim.time.time')
    def test_training_time_minutes_conversion(self, mock_time):
        """Test that minutes conversion is correct (seconds / 60)"""
        self.mock_args.target_acc = 0.5
        
        # Set up mock time: 120 seconds = 2 minutes
        start_time = 1000.0
        end_time = 1120.0
        expected_seconds = 120.0
        expected_minutes = expected_seconds / 60.0  # 2.0 minutes
        
        mock_time.return_value = end_time
        
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify the logged message contains correct seconds and minutes
        general_log_call = None
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'training time:' in args[0] and 'first time' not in args[0]:
                general_log_call = args[0]
                break
        
        self.assertIsNotNone(general_log_call)
        
        # Extract seconds and minutes from the log message
        seconds_match = re.search(r'training time: ([\d.]+) seconds', general_log_call)
        minutes_match = re.search(r'equivalent to ([\d.]+) minutes', general_log_call)
        
        self.assertIsNotNone(seconds_match)
        self.assertIsNotNone(minutes_match)
        
        logged_seconds = float(seconds_match.group(1))
        logged_minutes = float(minutes_match.group(1))
        
        # Verify the values match expected calculations
        self.assertAlmostEqual(logged_seconds, expected_seconds, places=2)
        self.assertAlmostEqual(logged_minutes, expected_minutes, places=2)
        # Verify minutes = seconds / 60
        self.assertAlmostEqual(logged_minutes, logged_seconds / 60.0, places=2)

    @patch('algorithms.engine.ffm_fedavg_depthffm_fim.time.time')
    def test_training_time_small_interval(self, mock_time):
        """Test training time calculation with a small time interval"""
        self.mock_args.target_acc = 0.5
        
        start_time = 1000.0
        end_time = 1000.123  # 0.123 seconds
        expected_seconds = 0.123
        expected_minutes = expected_seconds / 60.0
        
        mock_time.return_value = end_time
        
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Extract and verify time values
        general_log_call = None
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'training time:' in args[0] and 'first time' not in args[0]:
                general_log_call = args[0]
                break
        
        self.assertIsNotNone(general_log_call)
        seconds_match = re.search(r'training time: ([\d.]+) seconds', general_log_call)
        minutes_match = re.search(r'equivalent to ([\d.]+) minutes', general_log_call)
        
        logged_seconds = float(seconds_match.group(1))
        logged_minutes = float(minutes_match.group(1))
        
        self.assertAlmostEqual(logged_seconds, expected_seconds, places=3)
        self.assertAlmostEqual(logged_minutes, expected_minutes, places=6)

    @patch('algorithms.engine.ffm_fedavg_depthffm_fim.time.time')
    def test_training_time_large_interval(self, mock_time):
        """Test training time calculation with a large time interval"""
        self.mock_args.target_acc = 0.5
        
        start_time = 1000.0
        end_time = 3700.0  # 2700 seconds = 45 minutes
        expected_seconds = 2700.0
        expected_minutes = 45.0
        
        mock_time.return_value = end_time
        
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Extract and verify time values
        general_log_call = None
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'training time:' in args[0] and 'first time' not in args[0]:
                general_log_call = args[0]
                break
        
        self.assertIsNotNone(general_log_call)
        seconds_match = re.search(r'training time: ([\d.]+) seconds', general_log_call)
        minutes_match = re.search(r'equivalent to ([\d.]+) minutes', general_log_call)
        
        logged_seconds = float(seconds_match.group(1))
        logged_minutes = float(minutes_match.group(1))
        
        self.assertAlmostEqual(logged_seconds, expected_seconds, places=1)
        self.assertAlmostEqual(logged_minutes, expected_minutes, places=1)

    @patch('algorithms.engine.ffm_fedavg_depthffm_fim.time.time')
    def test_training_time_in_first_time_message(self, mock_time):
        """Test that training time is correctly included in the first time message"""
        self.mock_args.target_acc = 0.75
        
        start_time = 1000.0
        end_time = 1030.0  # 30 seconds
        expected_seconds = 30.0
        expected_minutes = 0.5
        
        mock_time.return_value = end_time
        
        best_test_acc = 0.8  # Above target
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Find the "first time" log message
        first_time_log_call = None
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'first time that best_test_acc' in args[0]:
                first_time_log_call = args[0]
                break
        
        self.assertIsNotNone(first_time_log_call)
        
        # Extract and verify time values from first time message
        seconds_match = re.search(r'training time: ([\d.]+) seconds', first_time_log_call)
        minutes_match = re.search(r'equivalent to ([\d.]+) minutes', first_time_log_call)
        
        self.assertIsNotNone(seconds_match)
        self.assertIsNotNone(minutes_match)
        
        logged_seconds = float(seconds_match.group(1))
        logged_minutes = float(minutes_match.group(1))
        
        self.assertAlmostEqual(logged_seconds, expected_seconds, places=1)
        self.assertAlmostEqual(logged_minutes, expected_minutes, places=2)

    @patch('algorithms.engine.ffm_fedavg_depthffm_fim.time.time')
    def test_training_time_consistency_between_messages(self, mock_time):
        """Test that training time is consistent between general and first time messages"""
        self.mock_args.target_acc = 0.75
        
        start_time = 1000.0
        end_time = 1050.0  # 50 seconds
        mock_time.return_value = end_time
        
        best_test_acc = 0.8  # Above target
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Extract time from both messages
        general_seconds = None
        first_time_seconds = None
        
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'training time:' in args[0]:
                seconds_match = re.search(r'training time: ([\d.]+) seconds', args[0])
                if seconds_match:
                    if 'first time' in args[0]:
                        first_time_seconds = float(seconds_match.group(1))
                    else:
                        general_seconds = float(seconds_match.group(1))
        
        # Both messages should have the same time value
        self.assertIsNotNone(general_seconds)
        self.assertIsNotNone(first_time_seconds)
        self.assertAlmostEqual(general_seconds, first_time_seconds, places=2)

    @patch('algorithms.engine.ffm_fedavg_depthffm_fim.time.time')
    def test_training_time_uses_current_time(self, mock_time):
        """Test that the function uses time.time() for end_time calculation"""
        self.mock_args.target_acc = 0.5
        
        start_time = 1000.0
        mock_end_time = 1015.25
        mock_time.return_value = mock_end_time
        
        best_test_acc = 0.6
        first_time_reaching_target_acc = False
        
        log_training_time(
            self.mock_args,
            start_time,
            best_test_acc,
            first_time_reaching_target_acc
        )
        
        # Verify time.time() was called
        self.assertTrue(mock_time.called)
        
        # Verify the calculated time matches
        expected_time = mock_end_time - start_time
        general_log_call = None
        for call in self.mock_logger.info.call_args_list:
            args, kwargs = call
            if len(args) > 0 and 'training time:' in args[0] and 'first time' not in args[0]:
                general_log_call = args[0]
                break
        
        self.assertIsNotNone(general_log_call)
        seconds_match = re.search(r'training time: ([\d.]+) seconds', general_log_call)
        logged_seconds = float(seconds_match.group(1))
        self.assertAlmostEqual(logged_seconds, expected_time, places=2)


if __name__ == '__main__':
    unittest.main()

