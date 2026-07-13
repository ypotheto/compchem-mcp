from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ypotheto_compchem_mcp.chemistry.parser import parse_qm_log_with_cclib


def test_parse_qm_log_with_cclib_success():
    mock_data = MagicMock()
    mock_data.scfenergies = [-2068.3] # in eV
    mock_data.atomnos = np.array([8, 1, 1]) # Water: O, H, H
    mock_data.atomcoords = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.95], [0.0, 0.95, 0.0]]])
    mock_data.moments = [None, [0.0, 0.0, 1.8]] # dipole in Debye
    mock_data.moenergies = [[-12.0, -10.0, 2.0, 4.0]] # MO energies
    mock_data.nocc = 2
    mock_data.atomcharges = {"mulliken": np.array([-0.4, 0.2, 0.2])}
    
    with patch("cclib.io.ccread", return_value=mock_data):
        res = parse_qm_log_with_cclib(Path("dummy.log"))
        assert res.ok is True
        assert res.energy_ev == -2068.3
        assert res.atom_symbols == ["O", "H", "H"]
        assert res.dipole_moment_debye == [0.0, 0.0, 1.8]
        assert res.homo_ev == -10.0
        assert res.lumo_ev == 2.0
        assert res.homo_lumo_gap_ev == 12.0
        assert len(res.mulliken_charges) == 3
        assert res.mulliken_charges[0].element == "O"
        assert res.mulliken_charges[0].charge == -0.4
        assert res.mulliken_charges[1].element == "H"
        assert res.mulliken_charges[1].charge == 0.2

def test_local_subprocess_runner():
    import sys
    import tempfile

    from ypotheto_compchem_mcp.chemistry.runner import LocalSubprocessRunner
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        runner = LocalSubprocessRunner()
        
        input_files = {
            "hello.py": "print('hello world')",
            "data.txt": "some test data"
        }
        
        result = runner.run_command(
            workspace_dir=tmp_path,
            job_id="test_job",
            command=[sys.executable, "hello.py"],
            input_files=input_files
        )
        
        assert result.return_code == 0
        assert "hello world" in result.stdout
        assert (tmp_path / "jobs" / "test_job" / "data.txt").read_text(encoding="utf-8") == "some test data"

def test_spaces_backend_mock():
    from unittest.mock import MagicMock, patch

    from ypotheto_compchem_mcp.storage import SpacesBackend
    
    mock_client = MagicMock()
    
    with patch("boto3.client", return_value=mock_client):
        backend = SpacesBackend(
            bucket="yp-mcp-bucket",
            endpoint_url="https://nyc3.digitaloceanspaces.com",
            access_key="test-key",
            secret_key="test-secret",
            region="nyc3",
            prefix="compchem-mcp"
        )
        
        # Test write_file
        backend.write_file("w1", "test.txt", b"hello spaces")
        mock_client.put_object.assert_called_once_with(
            Bucket="yp-mcp-bucket",
            Key="compchem-mcp/workspaces/w1/test.txt",
            Body=b"hello spaces"
        )
        
        # Test read_file
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = b"retrieved data"
        mock_client.get_object.return_value = mock_response
        
        data = backend.read_file("w1", "test.txt")
        assert data == b"retrieved data"
        mock_client.get_object.assert_called_once_with(
            Bucket="yp-mcp-bucket",
            Key="compchem-mcp/workspaces/w1/test.txt"
        )


