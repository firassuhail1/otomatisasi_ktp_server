<?php

namespace App\Http\Controllers;

use App\Models\DataKtp;
use Illuminate\Http\Request;

class DataKtpController extends Controller
{
    public function index()
    {
        $data = DataKtp::all();

        return response()->json(['success' => true, 'data' => $data], 200);
    }

    public function store(Request $request)
    {
        $validated = $request->validate([
            'nama' => 'required|string',
            'nik' => 'required|string',
            'alamat' => 'required|string',
            'ttl' => 'required|string',
            'jk' => 'required|string',
            'agama' => 'required|string',
        ]);

        // Simpan ke database
        DataKtp::create($validated);

        return response()->json(['success' => true, 'message' => 'Data berhasil disimpan']);
    }

    public function update(Request $request, $id)
    {
        $ktp = DataKTP::findOrFail($id);
        $ktp->update($request->all());
        return response()->json(['success' => true, 'message' => 'Updated']);
    }

    public function delete($id)
    {
        DataKtp::where('id', $id)->delete();

        return response()->json(['success' => true], 200);
    }
}
